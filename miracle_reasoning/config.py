"""Architecture adapters for the sibling Miracle training repositories."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

FAMILIES = ("flashnano-1.5", "flashnano-1.6", "flashnano-1.7", "flashlite-2.0",
            "flashlighting-2.0", "heavy-medium-2.0")


@dataclass(frozen=True)
class ModelConfig:
    family: str
    vocab_size: int
    hidden_size: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    ffn_dim: int
    max_positions: int
    window: int | None
    global_layers: tuple[int, ...] = ()
    global_kv_heads: int = 0
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    qk_norm: bool = True
    residual_scale: float = 1.0
    latent_dim: int = 0
    memory_slots: int = 0
    cbr_chunk_size: int = 0
    cbr_slots: int = 0
    cbr_norm_eps: float = 1e-6
    inner_loop_steps: int = 1
    eos_token_id: int = 2
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"Unsupported family {self.family!r}; expected one of {FAMILIES}. "
                             "Miracle Neural and architectures without a specification are excluded.")
        for key in ("vocab_size", "hidden_size", "layers", "query_heads", "kv_heads",
                    "head_dim", "ffn_dim", "max_positions", "inner_loop_steps"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0 or value >= 2**31:
                raise ValueError(f"{key} must be a positive int32")
        if self.head_dim % 2 or self.query_heads % self.kv_heads:
            raise ValueError("head_dim must be even and query_heads must divide into KV groups")
        if self.family != "heavy-medium-2.0" and self.hidden_size != self.query_heads*self.head_dim:
            raise ValueError("hidden_size must equal query_heads * head_dim")
        if self.window is not None and (type(self.window) is not int or self.window <= 0):
            raise ValueError("window must be positive or null")
        if self.global_kv_heads and (self.global_kv_heads < self.kv_heads or self.query_heads % self.global_kv_heads):
            raise ValueError("invalid global KV head count")
        if len(set(self.global_layers)) != len(self.global_layers) or any(type(i) is not int or i < 1 or i > self.layers for i in self.global_layers):
            raise ValueError("global layers must be unique, 1-based layer indices")
        for key in ("rope_theta", "norm_eps", "cbr_norm_eps", "residual_scale"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("latent_dim", "memory_slots", "cbr_chunk_size", "cbr_slots", "global_kv_heads"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        if not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id is outside vocabulary")
        if self.family == "flashnano-1.6" and (not self.cbr_chunk_size or not self.cbr_slots):
            raise ValueError("FlashNano 1.6 requires CBR chunk size and slots")
        if self.family in ("flashnano-1.5", "flashnano-1.6", "flashnano-1.7", "flashlighting-2.0") and not self.qk_norm:
            raise ValueError(f"{self.family} requires Q/K normalization")
        if self.family in ("flashlighting-2.0", "heavy-medium-2.0") and not self.latent_dim:
            raise ValueError("latent attention requires latent_dim")
        if self.family == "heavy-medium-2.0":
            s = self.source
            if not 1 <= s["router_top_k_min"] <= s["router_top_k_max"] <= s["routed_experts"]:
                raise ValueError("invalid router top-k")
            if s["shared_experts"] < 0 or s["router_groups"] < 1:
                raise ValueError("invalid expert/group count")
            if s["long_context"]["global_block_size"] <= 0 or s["long_context"]["global_slots"] < 0:
                raise ValueError("invalid sparse attention block geometry")

    def layer_kv_heads(self, index):
        return (self.global_kv_heads or self.kv_heads) if index+1 in self.global_layers else self.kv_heads

    def layer_window(self, index):
        return None if index+1 in self.global_layers else self.window

    def to_dict(self):
        return {"backend_format": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, data, family=None):
        if data.get("backend_format") == 1:
            values = {k: v for k, v in data.items() if k != "backend_format"}
            if family is not None and family != values["family"]:
                raise ValueError("family override conflicts with bundle")
            values["global_layers"] = tuple(values.get("global_layers", ()))
            return cls(**values)
        if family is None:
            kind = data.get("model_type", "")
            if "neural" in kind.lower():
                raise ValueError("Miracle Neural is outside this backend's scope")
            if "flashnano_17" in kind or "global_layer_indices_1based" in data:
                family = "flashnano-1.7"
            elif "cbr_chunk_size" in data:
                family = "flashnano-1.6"
            elif "flashnano" in kind:
                family = "flashnano-1.5"
            elif kind == "miracle_dense_2_7b" or "kv_latent_dim" in data:
                family = "flashlighting-2.0"
            elif "routed_experts" in data:
                family = "heavy-medium-2.0"
            elif kind == "miracle_dense":
                family = "flashlite-2.0"
            else:
                raise ValueError("Cannot identify architecture; specify --family explicitly")
        if family not in FAMILIES:
            raise ValueError(f"Unsupported family: {family}")
        def pick(*names, default=None):
            for name in names:
                if name in data:
                    return data[name]
            if default is not None:
                return default
            raise ValueError(f"Missing architecture field: {'/'.join(names)}")
        h = pick("hidden_size")
        layers = pick("layers", "num_hidden_layers", "num_layers")
        heads = pick("query_heads", "num_attention_heads")
        heavy = family == "heavy-medium-2.0"
        long = data.get("long_context", {})
        window = long.get("local_window") if heavy else next(
            (data[k] for k in ("window", "local_window", "local_window_size", "attention_window") if k in data), None)
        residual = data.get("residual_scale")
        if residual is None:
            residual = (2*layers)**-0.5 if family in ("flashlite-2.0", "flashlighting-2.0") else 1.0
        return cls(
            family=family, vocab_size=pick("vocab_size"), hidden_size=h, layers=layers,
            query_heads=heads, kv_heads=pick("kv_heads", "num_key_value_heads", "num_kv_heads"),
            head_dim=pick("head_dim", default=h//heads),
            ffn_dim=pick("ffn_dim", "intermediate_size", "expert_intermediate_size"),
            max_positions=long.get("target_context", 1_048_576) if heavy else pick("max_positions", "max_position_embeddings"),
            window=window, global_layers=tuple(data.get("global_layer_indices_1based", data.get("global_attention_layers", ()))),
            global_kv_heads=data.get("global_kv_heads", 0),
            rope_theta=long.get("rope_theta", 1e6) if heavy else data.get("rope_theta", 1e6),
            norm_eps=pick("norm_eps", "rms_norm_eps", "norm_epsilon", default=1e-6),
            tie_embeddings=pick("tie_embeddings", "tie_word_embeddings", default=True),
            qk_norm=data.get("use_qk_norm", True), residual_scale=residual,
            latent_dim=data.get("kv_latent_dim", data.get("latent_kv_dim", 0)),
            memory_slots=data.get("attention_memory_slots", 0),
            cbr_chunk_size=data.get("cbr_chunk_size", 0), cbr_slots=data.get("cbr_slots", 0),
            cbr_norm_eps=data.get("cbr_norm_eps", 1e-6), inner_loop_steps=data.get("inner_loop_steps", 1),
            eos_token_id=data.get("eos_token_id", 2), source=dict(data))

    def inverse_frequencies(self):
        dim, base, s = self.head_dim, self.rope_theta, self.source
        index = np.arange(0, dim, 2, dtype=np.float32) / dim
        inv = np.power(np.float32(base), -index)
        if self.family == "heavy-medium-2.0":
            split = int((dim//2) * s["long_context"].get("rope_local_fraction", 0.6))
            inv[split:] = np.power(np.float32(base*16), -index[split:])
        elif self.family == "flashlighting-2.0":
            kind, factor = s.get("rope_scaling_type", "yarn"), s.get("rope_scaling_factor", 8.0)
            if not math.isfinite(factor) or factor < 1:
                raise ValueError("rope_scaling_factor must be finite and >= 1")
            native = s.get("max_position_embeddings", self.max_positions)
            if kind == "linear":
                inv /= factor
            elif kind == "dynamic_ntk":
                if dim <= 2:
                    raise ValueError("dynamic_ntk requires head_dim > 2")
                ratio = factor*s.get("extension_max_position_embeddings", native)/native-(factor-1)
                inv = np.power(np.float32(base*max(ratio, 1)**(dim/(dim-2))), -index)
            elif kind == "yarn":
                low = native/s.get("rope_beta_fast", 32.0)
                high = native/s.get("rope_beta_slow", 1.0)
                if high <= low:
                    raise ValueError("YaRN requires beta_fast > beta_slow > 0")
                ramp = np.clip((2*np.pi/inv-low)/(high-low), 0, 1)
                inv = inv*(1-ramp)+(inv/factor)*ramp
            elif kind not in ("none", "native"):
                raise ValueError(f"Unsupported RoPE scaling: {kind}")
        elif self.family == "flashnano-1.7" and s.get("rope_scaling"):
            r = s["rope_scaling"]
            if r.get("type") != "yarn":
                raise ValueError("FlashNano 1.7 only supports YaRN rope_scaling")
            factor = float(r["factor"])
            if not math.isfinite(factor) or factor < 1:
                raise ValueError("YaRN factor must be finite and >= 1")
            def correction(rotations):
                return dim*math.log(r["orig_max_position"]/(rotations*2*math.pi))/(2*math.log(base))
            lo, hi = max(math.floor(correction(r["beta_fast"])), 0), min(math.ceil(correction(r["beta_slow"])), dim-1)
            if lo == hi:
                hi += 0.001
            ramp = np.clip((np.arange(dim//2, dtype=np.float32)-lo)/max(hi-lo, 1e-3), 0, 1)
            inv = (inv/factor)*(1-ramp)+inv*ramp
        return np.asarray(inv, dtype=np.float32)
