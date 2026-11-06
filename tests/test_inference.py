import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from miracle_reasoning import MiracleModel, ModelConfig
from miracle_reasoning.config import FAMILIES
from miracle_reasoning.weights import expected_shapes


def tiny(family="flashnano-1.5", tied=True):
    config = ModelConfig(family=family, vocab_size=32, hidden_size=16, layers=2,
                         query_heads=4, kv_heads=2, head_dim=4, ffn_dim=24, max_positions=32,
                         window=4, global_layers=(2,), tie_embeddings=tied)
    rng = np.random.default_rng(72)
    weights = {k: (rng.normal(0, 0.05, shape).astype(np.float32) if len(shape) == 2 else np.ones(shape, np.float32))
               for k, shape in expected_shapes(config).items()}
    return config, weights


@pytest.mark.parametrize("family", FAMILIES)
def test_native_streaming_parity(backend, tmp_path, family):
    root = Path(os.environ.get("MIRACLE_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
    if not (root/"Miracle 1.5 FlashNano").exists():
        pytest.skip("Set MIRACLE_SOURCE_ROOT to run parity against native training repositories")
    pytest.importorskip("torch")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(Path(__file__).with_name("make_reference.py")), str(root), family, str(tmp_path)]
    run = subprocess.run(command, text=True, capture_output=True, env=env, timeout=90)
    assert run.returncode == 0, run.stdout+run.stderr
    model = MiracleModel.from_pretrained(tmp_path, family=family, backend=backend, context_length=32)
    ids, reference = np.load(tmp_path/"ids.npy"), np.load(tmp_path/"reference.npy")
    actual = model.prefill(ids, return_all=True)
    np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-4)
    model.reset()
    chunked = np.concatenate([model.prefill(ids[:5], return_all=True), model.prefill(ids[5:], return_all=True)])
    np.testing.assert_allclose(chunked, actual, atol=2e-6, rtol=2e-5)
    if family == "flashnano-1.6":
        assert model.states[0]["ring"].length == 4
        assert model.states[0]["relay"].count == 8
    if family == "flashlighting-2.0":
        assert model.states[0]["latent"].shape[0] <= 6
    if family == "heavy-medium-2.0":
        assert len(model.states[0]["loops"]) == 2
        assert all(s["ring"].count == len(ids) for s in model.states[0]["loops"])


def test_cache_reset_bounds_and_chunking(backend):
    config, weights = tiny()
    model = MiracleModel(config, weights, backend=backend, context_length=12)
    ids = [1, 3, 5, 8, 2, 9, 11, 10, 9, 7, 6, 4]
    whole = model.prefill(ids, return_all=True)
    assert model.states[0]["ring"].length == 4
    assert model.states[1]["ring"].length == 12
    with pytest.raises(ValueError, match="context_length"):
        model.decode(1)
    assert model.position == 12
    model.reset()
    actual = np.stack([model.decode(i) for i in ids])
    np.testing.assert_array_equal(actual, whole)
    model.reset()
    for bad in ([32], [-1], [1.5], [[1]], [], [True]):
        with pytest.raises(ValueError):
            model.prefill(bad)
        assert model.position == 0


def test_weight_tiles_and_untied_head(backend):
    config, weights = tiny(tied=False)
    # Increase vocabulary so the 4 KiB weight budget forces output tiling.
    values = config.to_dict()
    values["vocab_size"] = 161
    config = ModelConfig.from_dict(values)
    rng = np.random.default_rng(4)
    weights["embed"] = rng.normal(0, .1, (161, 16)).astype(np.float32)
    weights["output"] = rng.normal(0, .1, (161, 16)).astype(np.float32)
    tiled = MiracleModel(config, weights, backend=backend, context_length=16, weight_cache_bytes=4096)
    full = MiracleModel(config, weights, backend=backend, context_length=16)
    np.testing.assert_allclose(tiled.prefill([1, 5, 8]), full.prefill([1, 5, 8]), atol=1e-7)
    assert tiled.weight_cache.used <= 4096


def test_generation_is_repeatable_and_validates_before_reset(backend):
    config, weights = tiny()
    model = MiracleModel(config, weights, backend=backend, context_length=12)
    a = model.generate([1, 3], max_new_tokens=4, temperature=.8, top_k=5, top_p=.9, seed=7)
    assert a == model.generate([1, 3], max_new_tokens=4, temperature=.8, top_k=5, top_p=.9, seed=7)
    position = model.position
    for kwargs in ({"max_new_tokens": -1}, {"max_new_tokens": 12}, {"temperature": float("nan")}, {"top_p": 0}):
        with pytest.raises(ValueError):
            model.generate([1, 3], **kwargs)
        assert model.position == position
    greedy = model.generate([1], max_new_tokens=1)[0]
    assert model.generate([1], max_new_tokens=4, eos_token_id=greedy) == [greedy]
    assert model.generate([1], max_new_tokens=0) == []


def test_bad_weights_rejected_before_device_allocation():
    config, weights = tiny()
    weights["layers.0.k"] = np.zeros((7, 16), np.float32)
    with pytest.raises(ValueError, match="expected"):
        MiracleModel(config, weights)
