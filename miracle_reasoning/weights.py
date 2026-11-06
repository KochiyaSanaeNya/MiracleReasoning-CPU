"""Safe, memory-mapped weights and explicit native checkpoint name adapters."""
from __future__ import annotations

import json
import math
import struct
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np


class SafeTensors(Mapping):
    """Read F32/F16/BF16 safetensors without PyTorch or a tokenizer dependency."""
    def __init__(self, path):
        self.path = Path(path)
        size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header")
            length, = struct.unpack("<Q", prefix)
            if length > min(100_000_000, size-8):
                raise ValueError("Invalid safetensors header length")
            metadata = json.loads(handle.read(length))
        self.offset = 8+length
        self.entries = {}
        intervals = []
        types = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}
        for name, entry in metadata.items():
            if name == "__metadata__":
                continue
            dtype = entry["dtype"]
            if dtype not in types:
                raise ValueError(f"Unsupported weight dtype {dtype} in {name}; export to FP32 first")
            shape, offsets = entry["shape"], entry["data_offsets"]
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f"Invalid shape for {name}")
            if len(offsets) != 2 or any(type(n) is not int for n in offsets):
                raise ValueError(f"Invalid offsets for {name}")
            start, end = offsets
            expected = math.prod(shape)*np.dtype(types[dtype]).itemsize
            if start < 0 or end-start != expected or end > size-self.offset:
                raise ValueError(f"Invalid safetensors extent for {name}")
            self.entries[name] = (tuple(shape), dtype, types[dtype], start)
            intervals.append((start, end))
        intervals.sort()
        if any(b[0] < a[1] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("Overlapping safetensors data")

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        shape, kind, dtype, start = self.entries[key]
        data = np.memmap(self.path, mode="r", offset=self.offset+start, dtype=dtype, shape=shape)
        return BF16Array(data) if kind == "BF16" else data


class BF16Array:
    """Keep BF16 mmap storage; expand only the selected weight tile to FP32."""
    def __init__(self, data):
        self.data, self.shape = data, data.shape

    def __getitem__(self, key):
        return (np.asarray(self.data[key], dtype=np.uint32) << 16).view(np.float32)

    def __array__(self, dtype=None, copy=None):
        result = self[:]
        return result.astype(dtype, copy=False) if dtype is not None else result


class ShardedSafeTensors(Mapping):
    def __init__(self, index):
        index = Path(index)
        self.names = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        self.shards = {}
        root = index.parent.resolve()
        for filename in set(self.names.values()):
            path = (root/filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Safetensors shard path escapes the model directory")
            self.shards[filename] = SafeTensors(path)
        for key, filename in self.names.items():
            if key not in self.shards[filename]:
                raise ValueError(f"Shard {filename} does not contain {key}")

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key):
        return self.shards[self.names[key]][key]


def save_safetensors(path, arrays):
    """Write FP32 tensors sequentially; no pickle and no full-model concatenation."""
    header, offset = {}, 0
    for key, value in arrays.items():
        length = math.prod(value.shape)*4
        header[key] = {"dtype": "F32", "shape": list(value.shape), "data_offsets": [offset, offset+length]}
        offset += length
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with Path(path).open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for value in arrays.values():
            # Convert/write at most 16 MiB at once, including BF16/FP16 sources.
            # This also keeps a multi-billion-row embedding out of a temporary
            # full-size FP32 array during export.
            if not value.shape:
                handle.write(np.asarray(value, dtype="<f4").tobytes())
                continue
            row_bytes = math.prod(value.shape[1:])*4
            tile = max(1, 16*1024*1024//max(row_bytes, 1))
            for start in range(0, value.shape[0], tile):
                array = np.asarray(value[start:start+tile], dtype="<f4", order="C")
                handle.write(memoryview(array).cast("B"))


class AliasedWeights(Mapping):
    def __init__(self, original, names):
        self.original, self.names = original, names
        missing = [v for v in names.values() if v not in original]
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {', '.join(missing[:12])}")

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key):
        return self.original[self.names[key]]


def native_names(config):
    c, names = config, {}
    if c.family in ("flashnano-1.5", "flashlite-2.0"):
        names["embed"] = "backbone." + ("token_embedding.weight" if c.family == "flashnano-1.5" else "embed_tokens.weight")
        names["norm"] = "backbone.norm.weight"
    else:
        names["embed"] = "embed_tokens.weight" if c.family == "flashlighting-2.0" else "embedding.weight"
        names["norm"] = "final_norm_weight" if c.family == "flashnano-1.7" else "final_norm.weight"
    if not c.tie_embeddings:
        names["output"] = "lm_head.weight"
    for i in range(c.layers):
        p = f"layers.{i}."
        if c.family in ("flashnano-1.5", "flashlite-2.0"):
            root = f"backbone.layers.{i}."
            an, fn, attn, ffn = "input_norm.weight", "post_attention_norm.weight", "attention.", "feed_forward."
        elif c.family == "flashnano-1.6":
            root = f"blocks.{i}."
            an, fn, attn, ffn = "attn_norm.weight", "ffn_norm.weight", "attn.", "ffn."
        elif c.family == "flashnano-1.7":
            root = f"blocks.{i}."
            an, fn, attn, ffn = "attn_norm_weight", "ffn_norm_weight", "attn.", "ffn."
        elif c.family == "flashlighting-2.0":
            root = f"layers.{i}."
            an, fn, attn, ffn = "input_norm.weight", None, "attention.", "feed_forward."
        else:
            root = f"blocks.{i}."
            an, fn, attn, ffn = "attention_norm.weight", "moe_norm.weight", "attention.", None
        names[p+"a_norm"] = root+an
        if fn:
            names[p+"f_norm"] = root+fn
        names[p+"q"] = root+attn+("w_q.weight" if c.family == "flashnano-1.6" else "q_proj.weight")
        names[p+"o"] = root+attn+("w_o.weight" if c.family == "flashnano-1.6" else "out_proj.weight" if c.family == "heavy-medium-2.0" else "o_proj.weight")
        if c.family in ("flashlighting-2.0", "heavy-medium-2.0"):
            names[p+"latent"] = root+attn+"kv_down.weight"
            if c.family == "flashlighting-2.0":
                for key, native in (("latent_norm", "kv_norm.weight"), ("k", "k_up.weight"), ("v", "v_up.weight")):
                    names[p+key] = root+attn+native
                names[p+"attention_gate"] = root+"attention_gate"
                names[p+"ffn_gate"] = root+"ffn_gate"
            else:
                names[p+"kv_up"] = root+attn+"kv_up.weight"
        else:
            for key in ("k", "v"):
                names[p+key] = root+attn+(f"w_{key}.weight" if c.family == "flashnano-1.6" else f"{key}_proj.weight")
        if c.qk_norm and c.family != "heavy-medium-2.0":
            for key in ("q", "k"):
                names[p+key+"_norm"] = root+attn+(f"qk_norm_{key}" if c.family == "flashnano-1.7" else f"{key}_norm.weight")
        if ffn:
            for key in ("gate", "up", "down"):
                names[p+"ffn."+key] = root+ffn+(f"w_{key}.weight" if c.family == "flashnano-1.6" else f"{key}_proj.weight")
        if c.family == "flashnano-1.6":
            names[p+"cbr"] = root+"cbr.queries"
        if c.family == "heavy-medium-2.0":
            for level in ("coarse", "fine"):
                for key in ("weight", "bias"):
                    names[p+f"router.{level}.{key}"] = root+f"moe.router.{level}.{key}"
            for key in ("group_bias", "expert_bias"):
                names[p+"router."+key] = root+"moe.router."+key
            for kind, count in (("experts", c.source["routed_experts"]), ("shared", c.source["shared_experts"])):
                for ei in range(count):
                    path = f"moe.experts.expert_{ei}." if kind == "experts" else f"moe.shared_experts.{ei}."
                    for key in ("gate", "up", "down"):
                        names[p+f"{kind}.{ei}.{key}"] = root+path+key+"_weight"
    return names


def expected_shapes(c):
    h, d, q, f = c.hidden_size, c.head_dim, c.query_heads*c.head_dim, c.ffn_dim
    result = {"embed": (c.vocab_size, h), "norm": (h,)}
    if not c.tie_embeddings:
        result["output"] = (c.vocab_size, h)
    for i in range(c.layers):
        p, kv = f"layers.{i}.", c.layer_kv_heads(i)*d
        result.update({p+"a_norm": (h,), p+"q": (q, h), p+"o": (h, q)})
        if c.family != "flashlighting-2.0":
            result[p+"f_norm"] = (h,)
        if c.latent_dim:
            result[p+"latent"] = (c.latent_dim, h)
        if c.family == "heavy-medium-2.0":
            result[p+"kv_up"] = (2*kv, c.latent_dim)
            e, g = c.source["routed_experts"], c.source["router_groups"]
            for level, n in (("coarse", g), ("fine", e)):
                result[p+f"router.{level}.weight"] = (n, h)
                result[p+f"router.{level}.bias"] = (n,)
            result[p+"router.group_bias"] = (g,)
            result[p+"router.expert_bias"] = (e,)
        else:
            inner = c.latent_dim or h
            result.update({p+"k": (kv, inner), p+"v": (kv, inner)})
            if c.qk_norm:
                result.update({p+"q_norm": (d,), p+"k_norm": (d,)})
        if c.family == "flashlighting-2.0":
            result.update({p+"latent_norm": (c.latent_dim,), p+"attention_gate": (1,), p+"ffn_gate": (1,)})
        if c.family == "flashnano-1.6":
            result[p+"cbr"] = (c.cbr_slots, h)
        roots = [p+"ffn."] if c.family != "heavy-medium-2.0" else [
            p+f"{kind}.{ei}." for kind, n in (("experts", c.source["routed_experts"]),
                                             ("shared", c.source["shared_experts"])) for ei in range(n)]
        for root in roots:
            result.update({root+"gate": (f, h), root+"up": (f, h), root+"down": (h, f)})
    return result


def adapt_weights(config, original):
    weights = original if "embed" in original else AliasedWeights(original, native_names(config))
    for key, shape in expected_shapes(config).items():
        if key not in weights:
            raise ValueError(f"Missing tensor {key}")
        if weights[key].shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {weights[key].shape}")
    return weights


class WeightCache:
    """Bound device weight residency and tile large vocabulary/linear matrices."""
    def __init__(self, backend, weights, budget_bytes):
        if type(budget_bytes) is not int or budget_bytes < 4096:
            raise ValueError("weight_cache_bytes must be an integer >= 4096")
        self.backend, self.weights, self.budget = backend, weights, budget_bytes
        self.cache, self.used = OrderedDict(), 0

    def tensor(self, name, start=None, stop=None):
        key = (name, start, stop)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        raw = self.weights[name]
        array = np.ascontiguousarray(raw if start is None else raw[start:stop], dtype=np.float32)
        while self.cache and self.used+array.nbytes > self.budget:
            _, old = self.cache.popitem(last=False)
            self.used -= old.size*4
        result = self.backend.upload(array)
        if array.nbytes <= self.budget:
            self.cache[key] = result
            self.used += array.nbytes
        return result

    def linear(self, x, name):
        rows, inner = self.weights[name].shape
        limit = min(self.budget, self.backend.device.max_allocation_bytes)
        tile = max(1, min(rows, limit//(inner*4)))
        if tile == rows:
            return self.backend.linear(x, self.tensor(name))
        # Model execution uses one token at a time, so vocabulary tiles can be
        # copied directly into the result without a transpose or host readback.
        if x.shape[0] != 1:
            raise ValueError("Tiled weight projection requires a single input row")
        result = self.backend.empty((1, rows))
        for start in range(0, rows, tile):
            stop = min(start+tile, rows)
            block = self.backend.linear(x, self.tensor(name, start, stop))
            self.backend.copy(block, result, target_offset=start)
        return result
