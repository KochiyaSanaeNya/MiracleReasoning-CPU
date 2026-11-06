"""Single-sequence inference with OpenCL tensor kernels and host orchestration."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .opencl import OpenCLBackend
from .weights import SafeTensors, WeightCache, adapt_weights


class _Ring:
    def __init__(self, backend, capacity, width):
        self.k = backend.empty((capacity, width))
        self.v = backend.empty((capacity, width))
        self.count, self.capacity, self.width = 0, capacity, width

    def append(self, backend, key, value):
        offset = (self.count % self.capacity)*self.width
        backend.copy(key, self.k, target_offset=offset)
        backend.copy(value, self.v, target_offset=offset)
        self.count += 1

    @property
    def length(self):
        return min(self.count, self.capacity)

    @property
    def start(self):
        return (self.count-self.length) % self.capacity


class MiracleModel:
    """FP32 next-token engine for six Miracle architectures.

    Prefill follows the same token-by-token path as decode. Each instance owns
    one sequence; call reset() for a new request. No training graph or Torch
    installation is required. Tokenization is intentionally a separate concern.
    """
    def __init__(self, config, weights, *, backend=None, context_length=None,
                 weight_cache_bytes=256*1024*1024):
        self.config = config
        if context_length is None:
            context_length = min(2048, config.max_positions)
        if type(context_length) is not int or not 0 < context_length <= config.max_positions:
            raise ValueError(f"context_length must be in [1, {config.max_positions}]")
        # Validate the full checkpoint before creating any device allocations.
        self.weights = adapt_weights(config, weights)
        self.backend = backend if backend is not None else OpenCLBackend()
        self.context_length = context_length
        self.weight_cache = WeightCache(self.backend, self.weights, weight_cache_bytes)
        self.frequencies = self.backend.upload(config.inverse_frequencies())
        self.reset()

    @classmethod
    def from_pretrained(cls, directory, *, family=None, **kwargs):
        directory = Path(directory)
        config = ModelConfig.from_dict(json.loads((directory/"config.json").read_text(encoding="utf-8")), family)
        if (directory/"model.safetensors").is_file():
            weights = SafeTensors(directory/"model.safetensors")
        elif (directory/"model.npz").is_file():
            weights = np.load(directory/"model.npz", allow_pickle=False)
        elif (directory/"model.safetensors.index.json").is_file():
            from .weights import ShardedSafeTensors
            weights = ShardedSafeTensors(directory/"model.safetensors.index.json")
        else:
            raise ValueError("Expected model.safetensors, a safetensors index, or model.npz; "
                             "use the export command for PyTorch checkpoints")
        return cls(config, weights, **kwargs)

    def reset(self):
        """Release sequence caches while retaining compiled kernels and weights."""
        self.position, self._failed = 0, False
        self.states = [{} for _ in range(self.config.layers)]

    def _norm(self, x, name, width=None, eps=None):
        c = self.config
        return self.backend.norm(x, self.weight_cache.tensor(name) if name else None,
                                 width or c.hidden_size, c.norm_eps if eps is None else eps)

    def _linear(self, x, name):
        return self.weight_cache.linear(x, name)

    def _ffn(self, x, root):
        return self._linear(self.backend.swiglu(self._linear(x, root+"gate"),
                                                self._linear(x, root+"up")), root+"down")

    def _rotate(self, x, positions, heads):
        return self.backend.rotate(x, self.frequencies, positions, heads, self.config.head_dim,
                                   self.config.family in ("flashnano-1.7", "flashlighting-2.0"))

    def _gqa(self, x, prefix, state, layer, position):
        b, c = self.backend, self.config
        heads = c.layer_kv_heads(layer)
        query = self._linear(x, prefix+"q")
        key, value = self._linear(x, prefix+"k"), self._linear(x, prefix+"v")
        if c.qk_norm:
            query = self._norm(query, prefix+"q_norm", c.head_dim)
            key = self._norm(key, prefix+"k_norm", c.head_dim)
        query, key = self._rotate(query, position, c.query_heads), self._rotate(key, position, heads)
        if "ring" not in state:
            capacity = min(c.layer_window(layer) or self.context_length, self.context_length)
            state["ring"] = _Ring(b, capacity, heads*c.head_dim)
        ring = state["ring"]
        ring.append(b, key, value)
        relay = state.get("relay")
        attended = b.attention(query, ring.k, ring.v, heads=c.query_heads, kv_heads=heads,
                               dim=c.head_dim, start=ring.start, length=ring.length,
                               extra_keys=relay.k if relay else None, extra_values=relay.v if relay else None,
                               extra_length=relay.count if relay else 0)
        if c.family == "flashnano-1.6":
            self._update_cbr(x, prefix, state, heads)
        return self._linear(attended, prefix+"o")

    def _update_cbr(self, x, prefix, state, heads):
        b, c = self.backend, self.config
        if "pending" not in state:
            state["pending"] = b.empty((min(c.cbr_chunk_size, self.context_length), c.hidden_size))
        offset = self.position % c.cbr_chunk_size
        b.copy(x, state["pending"], target_offset=offset*c.hidden_size)
        if offset+1 != c.cbr_chunk_size:
            return
        queries = self._norm(self.weight_cache.tensor(prefix+"cbr"), None,
                             c.hidden_size, c.cbr_norm_eps)
        scores = b.empty((c.cbr_slots, c.cbr_chunk_size))
        b.run("pool_scores", (c.cbr_chunk_size, c.cbr_slots), state["pending"], queries,
              scores, c.cbr_chunk_size, c.cbr_slots, c.hidden_size)
        probs = b.softmax(scores)
        pooled, anchors = b.empty((c.cbr_slots, c.hidden_size)), b.empty((c.cbr_slots,))
        b.run("pool_mix", (c.hidden_size, c.cbr_slots), state["pending"], probs, pooled, anchors,
              c.cbr_chunk_size, c.cbr_slots, c.hidden_size, self.position+1-c.cbr_chunk_size)
        key, value = self._linear(pooled, prefix+"k"), self._linear(pooled, prefix+"v")
        key = self._rotate(self._norm(key, prefix+"k_norm", c.head_dim), anchors, heads)
        if "relay" not in state:
            capacity = (self.context_length//c.cbr_chunk_size)*c.cbr_slots
            state["relay"] = _Ring(b, capacity, heads*c.head_dim)
        relay = state["relay"]
        count = c.cbr_slots*heads*c.head_dim
        b.copy(key, relay.k, target_offset=relay.count*heads*c.head_dim, count=count)
        b.copy(value, relay.v, target_offset=relay.count*heads*c.head_dim, count=count)
        relay.count += c.cbr_slots

    def _lighting(self, x, prefix, state, position):
        b, c = self.backend, self.config
        query = self._rotate(self._norm(self._linear(x, prefix+"q"), prefix+"q_norm", c.head_dim),
                             position, c.query_heads)
        latent = self._norm(self._linear(x, prefix+"latent"), prefix+"latent_norm", c.latent_dim)
        old = state.get("latent")
        n = 0 if old is None else old.shape[0]
        latents = b.empty((n+1, c.latent_dim))
        if old is not None:
            b.copy(old, latents)
        b.copy(latent, latents, target_offset=n*c.latent_dim)
        positions = state.get("positions", [])+[self.position]
        counts = state.get("counts", [])+[1.0]
        key = self._norm(self._linear(latents, prefix+"k"), prefix+"k_norm", c.head_dim)
        key_positions = b.upload(np.asarray(positions, dtype=np.float32))
        key = self._rotate(key, key_positions, c.kv_heads)
        value = self._linear(latents, prefix+"v")
        scores = b.empty((c.query_heads, n+1))
        b.run("scores", (n+1, c.query_heads), query, key, key, scores, c.query_heads,
              c.kv_heads, c.head_dim, n+1, 0, n+1, 0)
        b.run("latent_mask", (n+1, c.query_heads), scores, key_positions, c.query_heads,
              n+1, state.get("memory_size", 0), self.position-c.window+1)
        attended = b.mix(b.softmax(scores), value, c.query_heads, c.kv_heads, c.head_dim, 0, n+1)
        # A decode appends one row. torch.tensor_split then either keeps every
        # old row or merges the first two; weights/rounded anchors are retained.
        old_count = max(0, n+1-c.window)
        if c.memory_slots == 0 and old_count:
            compact = b.empty((n+1-old_count, c.latent_dim))
            b.copy(latents, compact, source_offset=old_count*c.latent_dim)
            latents, positions, counts = compact, positions[old_count:], counts[old_count:]
            memory_size = 0
        elif old_count > c.memory_slots:
            compact = b.empty((n, c.latent_dim))
            a, z = np.float32(counts[0]), np.float32(counts[1])
            b.run("merge_rows", (c.latent_dim,), latents, compact, c.latent_dim, float(a), float(z))
            b.copy(latents, compact, source_offset=2*c.latent_dim,
                   target_offset=c.latent_dim, count=(n-1)*c.latent_dim)
            mean = np.float32(np.float32(positions[0])*a+np.float32(positions[1])*z)/np.float32(a+z)
            positions = [int(np.rint(mean))]+positions[2:]
            counts = [float(a+z)]+counts[2:]
            latents, memory_size = compact, c.memory_slots
        else:
            memory_size = old_count
        state.update(latent=latents, positions=positions, counts=counts, memory_size=memory_size)
        return self._linear(attended, prefix+"o")

    def _heavy_attention(self, x, prefix, state, position):
        b, c = self.backend, self.config
        query = self._norm(self._linear(x, prefix+"q"), None, c.head_dim, 1e-6)
        kv = self._linear(self._linear(x, prefix+"latent"), prefix+"kv_up")
        width = c.kv_heads*c.head_dim
        key, value = b.empty((1, width)), b.empty((1, width))
        b.copy(kv, key, count=width)
        b.copy(kv, value, source_offset=width, count=width)
        query = self._rotate(query, position, c.query_heads)
        key = self._rotate(self._norm(key, None, c.head_dim, 1e-6), position, c.kv_heads)
        if "ring" not in state:
            state["ring"] = _Ring(b, self.context_length, width)
        ring = state["ring"]
        ring.append(b, key, value)
        length = min(c.window, ring.count)
        start = ring.count-length
        long = c.source["long_context"]
        block = long["global_block_size"]
        blocks = (ring.count+block-1)//block
        mean_keys, mean_values = b.empty((blocks, width)), b.empty((blocks, width))
        for src, dst in ((ring.k, mean_keys), (ring.v, mean_values)):
            b.run("block_means", (width, blocks), src, dst, ring.count, width, block)
        scores = b.empty((c.query_heads, length+blocks))
        b.run("scores", (length+blocks, c.query_heads), query, ring.k, mean_keys, scores,
              c.query_heads, c.kv_heads, c.head_dim, ring.capacity, start, length, blocks)
        # Top-k is control flow; projections, dot products, pooling, softmax,
        # expert MLPs and weighted value reductions stay on the OpenCL device.
        host = b.read(scores)
        end = np.minimum((np.arange(blocks)+1)*block-1, self.position)
        host[:, length:] -= np.float32(long.get("alibi_slope_scale", 0.015)/math.sqrt(c.head_dim))*np.log1p(
            (self.position-end).astype(np.float32))
        slots = min(long["global_slots"], blocks)
        rank = np.argsort(-host[:, length:], axis=1, kind="stable")
        for h in range(c.query_heads):
            host[h, length+rank[h, slots:]] = -np.inf
        attended = b.mix(b.softmax(b.upload(host)), ring.v, c.query_heads, c.kv_heads, c.head_dim,
                          start, length, mean_values, blocks)
        return self._linear(attended, prefix+"o")

    def _moe(self, x, prefix):
        b, c, s = self.backend, self.config, self.config.source
        coarse = b.read(self._linear(x, prefix+"router.coarse.weight"))[0]
        fine = b.read(self._linear(x, prefix+"router.fine.weight"))[0]
        coarse += np.asarray(self.weights[prefix+"router.coarse.bias"])+np.asarray(self.weights[prefix+"router.group_bias"])
        fine += np.asarray(self.weights[prefix+"router.fine.bias"])+np.asarray(self.weights[prefix+"router.expert_bias"])
        per_group = (s["routed_experts"]+s["router_groups"]-1)//s["router_groups"]
        groups = np.minimum(np.arange(s["routed_experts"])//per_group, s["router_groups"]-1)
        logits = (fine+coarse[groups])/np.float32(max(s.get("router_temperature", 1.0), 1e-4))
        probs = b.read(b.softmax(b.upload(logits.reshape(1, -1))))[0]
        max_k, min_k = s["router_top_k_max"], s["router_top_k_min"]
        indices = np.argsort(-probs, kind="stable")[:max_k]
        selected = probs[indices].copy()
        confidence = selected[0]-selected[min(max_k-1, 1)]
        entropy = -(probs*np.log(np.maximum(probs, np.float32(1e-9)))).sum()
        width_score = np.rint((np.float32(1)-confidence)*np.float32(2)+entropy/max(np.float32(np.log(s["routed_experts"])), np.float32(1)))
        active = int(np.clip(min_k+width_score, min_k, max_k))
        selected[active:] = 0
        selected /= max(selected.sum(), np.float32(1e-9))
        result = b.upload(np.zeros(x.shape, dtype=np.float32))
        for ei, weight in zip(indices[:active], selected[:active]):
            result = b.add(result, self._ffn(x, prefix+f"experts.{ei}."), float(weight))
        for ei in range(s["shared_experts"]):
            result = b.add(result, self._ffn(x, prefix+f"shared.{ei}."), c.residual_scale/max(s["shared_experts"], 1))
        return result

    def _step(self, token):
        b, c = self.backend, self.config
        hidden = b.upload(np.asarray(self.weights["embed"][token:token+1], dtype=np.float32))
        position = b.upload(np.asarray([self.position], dtype=np.float32))
        for layer, state in enumerate(self.states):
            prefix = f"layers.{layer}."
            if c.family == "heavy-medium-2.0":
                if "loops" not in state:
                    state["loops"] = [{} for _ in range(c.inner_loop_steps)]
                scale = c.residual_scale/math.sqrt(c.inner_loop_steps)
                for inner in state["loops"]:
                    attended = self._heavy_attention(self._norm(hidden, prefix+"a_norm"), prefix, inner, position)
                    hidden = b.add(hidden, attended, scale)
                    hidden = b.add(hidden, self._moe(self._norm(hidden, prefix+"f_norm"), prefix), scale)
            elif c.family == "flashlighting-2.0":
                normalized = self._norm(hidden, prefix+"a_norm")
                attended = self._lighting(normalized, prefix, state, position)
                ag = float(np.asarray(self.weights[prefix+"attention_gate"])[0])
                fg = float(np.asarray(self.weights[prefix+"ffn_gate"])[0])
                # Both branches consume the same pre-normalized input.
                residual = b.add(b.upload(np.zeros(hidden.shape, dtype=np.float32)), attended, ag)
                residual = b.add(residual, self._ffn(normalized, prefix+"ffn."), fg)
                hidden = b.add(hidden, residual, c.residual_scale)
            else:
                attended = self._gqa(self._norm(hidden, prefix+"a_norm"), prefix, state, layer, position)
                hidden = b.add(hidden, attended, c.residual_scale)
                hidden = b.add(hidden, self._ffn(self._norm(hidden, prefix+"f_norm"), prefix+"ffn."), c.residual_scale)
        self.position += 1
        return self._norm(hidden, "norm")

    def _tokens(self, ids):
        array = np.asarray(ids)
        if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
            raise ValueError("input_ids must be a non-empty 1D integer sequence")
        if np.any(array < 0) or np.any(array >= self.config.vocab_size):
            raise ValueError("input token is outside the vocabulary")
        return array

    def prefill(self, input_ids, *, return_all=False):
        """Append a prompt, returning [vocab] or, if requested, [tokens,vocab]."""
        tokens = self._tokens(input_ids)
        if self._failed:
            raise RuntimeError("A previous device operation failed; call reset() before reusing this model")
        if self.position+len(tokens) > self.context_length:
            raise ValueError("input would exceed context_length; reset or allocate a larger context")
        result = []
        try:
            for i, token in enumerate(tokens):
                hidden = self._step(int(token))
                if return_all or i == len(tokens)-1:
                    logits = self._linear(hidden, "embed" if self.config.tie_embeddings else "output")
                    result.append(self.backend.read(logits)[0])
        except Exception:
            self._failed = True
            raise
        return np.stack(result) if return_all else result[0]

    def decode(self, token_id):
        """Append one token to the current sequence and return next-token logits."""
        return self.prefill([token_id])

    def generate(self, input_ids, *, max_new_tokens=32, temperature=0.0,
                 top_k=0, top_p=1.0, seed=17, eos_token_id=None):
        """Return generated token IDs, excluding the prompt and including EOS."""
        tokens = self._tokens(input_ids)
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if type(top_k) is not int or top_k < 0 or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_k must be non-negative and top_p must be in (0,1]")
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        if type(eos) is not int or not 0 <= eos < self.config.vocab_size:
            raise ValueError("eos_token_id is outside vocabulary")
        if len(tokens)+max(0, max_new_tokens-1) > self.context_length:
            raise ValueError("prompt plus requested generation exceeds context_length")
        self.reset()
        if max_new_tokens == 0:
            return []
        rng, generated = np.random.default_rng(seed), []
        logits = self.prefill(tokens)
        for i in range(max_new_tokens):
            if not np.all(np.isfinite(logits)):
                raise FloatingPointError("non-finite output logits")
            if temperature == 0:
                token = int(np.argmax(logits))
            else:
                order = np.argsort(-logits, kind="stable")
                if top_k:
                    order = order[:top_k]
                scores = logits[order].astype(np.float64)/temperature
                probs = np.exp(scores-scores.max())
                probs /= probs.sum()
                cutoff = min(len(order), int(np.searchsorted(np.cumsum(probs), top_p))+1)
                order, probs = order[:cutoff], probs[:cutoff]
                token = int(rng.choice(order, p=probs/probs.sum()))
            generated.append(token)
            if token == eos or i+1 == max_new_tokens:
                break
            logits = self.decode(token)
        return generated
