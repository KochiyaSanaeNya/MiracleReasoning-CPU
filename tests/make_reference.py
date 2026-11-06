"""Run a sibling training implementation in isolation, avoiding package collisions."""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch


def main():
    root, family, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    folder = {"flashnano-1.5": "Miracle 1.5 FlashNano", "flashnano-1.6": "Miracle 1.6 FlashNano",
              "flashnano-1.7": "Miracle 1.7 FlashNano/code", "flashlite-2.0": "Miracle 2.0FlashLite",
              "flashlighting-2.0": "Miracle 2.0FlashLighting", "heavy-medium-2.0": "Miracle 2.0 Heavy Medium"}[family]
    sys.path.insert(0, str(root/folder))
    torch.set_num_threads(1)
    torch.manual_seed(31)
    if family == "flashnano-1.5":
        from miracle_flashnano import FlashNanoConfig, MiracleFlashNanoForCausalLM
        config = FlashNanoConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                                 intermediate_size=24, max_position_embeddings=32,
                                 local_window_size=4, global_attention_layers=(2,))
        model = MiracleFlashNanoForCausalLM(config)
    elif family == "flashnano-1.6":
        from miracle_dense.config import DenseModelConfig
        from miracle_dense.model.model import DenseLM
        config = DenseModelConfig(name="micro", vocab_size=32, hidden_size=16, layers=2,
                                  query_heads=4, kv_heads=2, head_dim=4, ffn_dim=24,
                                  max_positions=32, max_relay_tokens=64, window=4,
                                  cbr_chunk_size=3, cbr_slots=2, activation_dtype="float32")
        model = DenseLM(config)
    elif family == "flashnano-1.7":
        from config import TestModelConfig
        from model.model import DenseLM
        config = TestModelConfig(vocab_size=32, hidden_size=16, layers=2, query_heads=4,
                                 kv_heads=2, global_kv_heads=4, head_dim=4, ffn_dim=24,
                                 local_window=4, max_positions=32, global_layer_indices_1based=(2,))
        model = DenseLM(config)
    elif family == "flashlite-2.0":
        from miracle_dense.config import MiracleDenseConfig
        from miracle_dense.model.model import MiracleDenseForCausalLM
        config = MiracleDenseConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                                    num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                                    intermediate_size=24, max_position_embeddings=32,
                                    native_context_length=32, extension_context_length=64, attention_window=4)
        model = MiracleDenseForCausalLM(config)
    elif family == "flashlighting-2.0":
        from model.config import DenseModelConfig
        from model.model import DenseCausalLM
        config = DenseModelConfig(vocab_size=32, hidden_size=16, num_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, kv_latent_dim=8,
                                  intermediate_size=24, max_position_embeddings=32,
                                  native_context_length=32, extension_max_position_embeddings=64,
                                  attention_window=4, attention_memory_slots=2, attention_prefill_chunk_size=1)
        model = DenseCausalLM(config, use_optimized_kernels=False)
        with torch.no_grad():
            for layer in model.layers:
                layer.attention_gate.fill_(0.7)
                layer.ffn_gate.fill_(1.2)
    else:
        from m2hm.spec import ModelConfig, LongContextConfig
        from m2hm.model import FrontierMoEModel
        from m2hm.backend import TorchBackend
        config = replace(ModelConfig(), vocab_size=32, hidden_size=16, num_layers=2,
                         num_attention_heads=4, num_kv_heads=2, head_dim=4, latent_kv_dim=8,
                         routed_experts=4, shared_experts=1, expert_intermediate_size=12,
                         router_groups=2, router_top_k_min=1, router_top_k_max=2, mtp_depth=0,
                         long_context=replace(LongContextConfig(), target_context=32,
                                              local_window=4, global_block_size=3, global_slots=2))
        model = FrontierMoEModel(config, backend=TorchBackend(device="cpu", dtype=torch.float32),
                                device="cpu", dtype=torch.float32)
        # Upstream _local_window returns [1,1,T,W] validity while _global_blocks
        # returns [B,H,T,G]; their torch.cat fails for H>1. Expand only the
        # broadcast mask in this isolated oracle, leaving arithmetic unchanged.
        for layer in model.blocks:
            original = layer.attention._local_window
            def expanded_local(*args, _original=original, **kwargs):
                scores, values, valid = _original(*args, **kwargs)
                return scores, values, valid.expand_as(scores)
            layer.attention._local_window = expanded_local
    model.eval()
    ids = [1, 5, 3, 7, 11, 9, 4, 17, 8, 6, 10, 12]
    cache, cbr, collected = None, None, []
    if family == "flashnano-1.7":
        from model.cache import CacheManager
        cache = CacheManager(config, batch_size=1)
    if family == "heavy-medium-2.0":
        from m2hm.attention import LatentKVCache
        cache = [LatentKVCache() for _ in range(config.num_layers)]
    with torch.no_grad():
        for i, token in enumerate(ids):
            x = torch.tensor([[token]])
            if family == "flashnano-1.7":
                out = model(x, torch.tensor([[i]]), torch.zeros_like(x), torch.ones_like(x, dtype=torch.bool),
                            cache=cache, use_cache=True, logits_mode="all")
            elif family == "heavy-medium-2.0":
                out = model(x, caches=cache, positions=torch.tensor([i]), output_mtp=False)
            elif family == "flashnano-1.6":
                out = model(x, past_key_values=cache, cbr_state=cbr, use_cache=True)
                cache, cbr = out.past_key_values, out.cbr_state
            else:
                out = model(x, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
            collected.append(out.logits[0, -1].float().numpy())
    output.mkdir(parents=True, exist_ok=True)
    (output/"config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    np.savez(output/"model.npz", **{k: v.detach().float().numpy() for k, v in model.state_dict().items()})
    np.save(output/"reference.npy", np.stack(collected))
    np.save(output/"ids.npy", np.array(ids, dtype=np.int32))


if __name__ == "__main__":
    main()
