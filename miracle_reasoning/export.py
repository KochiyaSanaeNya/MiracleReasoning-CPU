"""Convert native weights into a standalone, pickle-free inference bundle."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .weights import SafeTensors, ShardedSafeTensors, adapt_weights, save_safetensors


class _TorchArray:
    def __init__(self, tensor):
        self.tensor, self.shape = tensor, tuple(tensor.shape)

    def __getitem__(self, key):
        return self.tensor[key].detach().cpu().float().numpy()

    def __array__(self, dtype=None, copy=None):
        result = self.tensor.detach().cpu().float().numpy()
        return result.astype(dtype, copy=False) if dtype is not None else result


class _TorchWeights(Mapping):
    def __init__(self, tensors):
        self.tensors = tensors

    def __iter__(self):
        return iter(self.tensors)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, key):
        return _TorchArray(self.tensors[key])


def export_checkpoint(checkpoint, config_path, output, *, family=None):
    config_path, checkpoint, output = Path(config_path), Path(checkpoint), Path(output)
    if config_path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        values = json.loads(config_path.read_text(encoding="utf-8"))
    config = ModelConfig.from_dict(values, family)
    if checkpoint.suffix == ".safetensors":
        original = SafeTensors(checkpoint)
    elif checkpoint.name.endswith(".safetensors.index.json"):
        original = ShardedSafeTensors(checkpoint)
    elif checkpoint.suffix == ".npz":
        original = np.load(checkpoint, allow_pickle=False)
    else:
        import torch
        # Never execute arbitrary pickle objects from an imported checkpoint.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        for key in ("state_dict", "model_state_dict", "parameters", "model"):
            if isinstance(payload, dict) and isinstance(payload.get(key), dict):
                payload = payload[key]
                break
        if not isinstance(payload, dict) or not all(isinstance(v, torch.Tensor) for v in payload.values()):
            raise ValueError("Checkpoint must contain a tensor state_dict (or parameters/model wrapper)")
        original = _TorchWeights(payload)
    canonical = adapt_weights(config, original)
    output.mkdir(parents=True, exist_ok=True)
    weights_tmp, config_tmp = output/"model.safetensors.tmp", output/"config.json.tmp"
    # Do not replace a file that is still mapped as the input.
    if checkpoint.resolve() == (output/"model.safetensors").resolve():
        raise ValueError("Export output must differ from the input checkpoint path")
    try:
        save_safetensors(weights_tmp, canonical)
        config_tmp.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        os.replace(weights_tmp, output/"model.safetensors")
        os.replace(config_tmp, output/"config.json")
    finally:
        weights_tmp.unlink(missing_ok=True)
        config_tmp.unlink(missing_ok=True)
    return config
