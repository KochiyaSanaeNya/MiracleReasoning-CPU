import json
import struct
from dataclasses import replace

import numpy as np
import pytest

from miracle_reasoning import MiracleModel, ModelConfig
from miracle_reasoning.cli import main
from miracle_reasoning.export import export_checkpoint
from miracle_reasoning.weights import SafeTensors, ShardedSafeTensors, native_names, save_safetensors
from test_inference import tiny


def test_safetensors_roundtrip_and_bfloat16_tiles(tmp_path):
    arrays = {"x": np.arange(21, dtype=np.float32).reshape(3, 7), "scale": np.ones(7, np.float32)}
    save_safetensors(tmp_path/"f32.safetensors", arrays)
    mapped = SafeTensors(tmp_path/"f32.safetensors")
    for name, expected in arrays.items():
        np.testing.assert_array_equal(mapped[name], expected)
    bits = (arrays["x"].view(np.uint32) >> 16).astype("<u2")
    header = json.dumps({"x": {"dtype": "BF16", "shape": [3, 7], "data_offsets": [0, bits.nbytes]}}).encode()
    (tmp_path/"bf16.safetensors").write_bytes(struct.pack("<Q", len(header))+header+bits.tobytes())
    bf16 = SafeTensors(tmp_path/"bf16.safetensors")["x"]
    np.testing.assert_array_equal(bf16[1:2], arrays["x"][1:2])
    np.testing.assert_array_equal(np.asarray(bf16), arrays["x"])


@pytest.mark.parametrize("payload", [b"", struct.pack("<Q", 1024)+b"{}",
    struct.pack("<Q", 62)+b'{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'])
def test_truncated_weights_rejected(tmp_path, payload):
    path = tmp_path/"bad.safetensors"
    path.write_bytes(payload)
    with pytest.raises((ValueError, KeyError)):
        SafeTensors(path)


def test_shards_and_path_containment(tmp_path):
    save_safetensors(tmp_path/"part.safetensors", {"x": np.ones((3,), np.float32)})
    index = tmp_path/"model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "part.safetensors"}}))
    np.testing.assert_array_equal(ShardedSafeTensors(index)["x"], np.ones(3))
    index.write_text(json.dumps({"weight_map": {"x": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="escapes"):
        ShardedSafeTensors(index)


def test_export_native_npz_to_standalone_bundle(backend, tmp_path, capsys):
    config, weights = tiny()
    source, output = tmp_path/"native", tmp_path/"bundle"
    source.mkdir()
    names = native_names(config)
    np.savez(source/"model.npz", **{names[k]: v for k, v in weights.items()})
    (source/"config.json").write_text(json.dumps(config.to_dict()))
    exported = export_checkpoint(source/"model.npz", source/"config.json", output)
    assert exported == config
    restored = MiracleModel.from_pretrained(output, backend=backend)
    original = MiracleModel(config, weights, backend=backend)
    np.testing.assert_array_equal(restored.prefill([1, 5, 7]), original.prefill([1, 5, 7]))
    assert main(["inspect", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["family"] == config.family


def test_config_roundtrip_and_validation():
    config, _ = tiny()
    assert ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    for changes in ({"head_dim": 3}, {"query_heads": 3}, {"layers": True}, {"window": 0},
                    {"norm_eps": float("nan")}, {"global_layers": (0,)}, {"family": "miracle-neural"}):
        with pytest.raises(ValueError):
            replace(config, **changes)
    with pytest.raises(ValueError, match="scope"):
        ModelConfig.from_dict({"model_type": "miracle_neural"})
