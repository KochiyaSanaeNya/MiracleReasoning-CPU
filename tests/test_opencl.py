import numpy as np
import pytest

from miracle_reasoning import OpenCLBackend, OpenCLError, list_devices


def test_cpu_selection_never_falls_back_to_gpu():
    try:
        devices = list_devices()
    except OpenCLError:
        pytest.skip("No OpenCL loader")
    if any(d.kind == "cpu" for d in devices):
        assert OpenCLBackend().device.kind == "cpu"
    else:
        with pytest.raises(OpenCLError, match="No matching OpenCL CPU"):
            OpenCLBackend()


def test_opencl_primitives_with_non_aligned_shapes(backend):
    b = backend
    rng = np.random.default_rng(9)
    x = rng.normal(size=(3, 19)).astype(np.float32)
    w = rng.normal(size=(7, 19)).astype(np.float32)
    np.testing.assert_allclose(b.read(b.linear(b.upload(x), b.upload(w))), x@w.T, atol=4e-6)
    scales = rng.normal(size=19).astype(np.float32)
    expected = x/np.sqrt(np.mean(x*x, axis=1, keepdims=True)+1e-6)*scales
    np.testing.assert_allclose(b.read(b.norm(b.upload(x), b.upload(scales), 19, 1e-6)), expected, atol=1e-6)
    logits = np.array([[1000, 1001, -1000], [-1000, -1001, -1002]], np.float32)
    e = np.exp(logits-logits.max(axis=1, keepdims=True))
    np.testing.assert_allclose(b.read(b.softmax(b.upload(logits))), e/e.sum(axis=1, keepdims=True), atol=1e-7)


@pytest.mark.parametrize("split", [False, True])
def test_rope_layouts(backend, split):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 3, 6)).astype(np.float32)
    pos = np.array([7, 29], np.float32)
    freq = np.array([1, .1, .01], np.float32)
    angles = pos[:, None, None]*freq[None, None, :]
    first, second = (x[..., :3], x[..., 3:]) if split else (x[..., ::2], x[..., 1::2])
    u, v = first*np.cos(angles)-second*np.sin(angles), first*np.sin(angles)+second*np.cos(angles)
    expected = np.concatenate([u, v], axis=-1) if split else np.stack([u, v], axis=-1).reshape(2, 3, 6)
    actual = backend.rotate(backend.upload(x.reshape(2, 18)), backend.upload(freq), backend.upload(pos), 3, 6, split)
    np.testing.assert_allclose(backend.read(actual).reshape(x.shape), expected, atol=3e-6)


def test_attention_group_mapping_ring_and_relays(backend):
    b, rng = backend, np.random.default_rng(6)
    q = rng.normal(size=(4, 6)).astype(np.float32)
    k, v = [rng.normal(size=(5, 2, 6)).astype(np.float32) for _ in range(2)]
    rk, rv = [rng.normal(size=(3, 2, 6)).astype(np.float32) for _ in range(2)]
    # Logical keys wrap from physical rows [3,4,0,1]; relays are read once.
    keys, values = np.concatenate([k[[3, 4, 0, 1]], rk]), np.concatenate([v[[3, 4, 0, 1]], rv])
    expected = []
    for head in range(4):
        logits = keys[:, head//2]@q[head]/np.sqrt(6)
        probs = np.exp(logits-logits.max())
        probs /= probs.sum()
        expected.append(probs@values[:, head//2])
    actual = b.attention(b.upload(q.reshape(1, -1)), b.upload(k.reshape(5, -1)), b.upload(v.reshape(5, -1)),
                         heads=4, kv_heads=2, dim=6, start=3, length=4,
                         extra_keys=b.upload(rk.reshape(3, -1)), extra_values=b.upload(rv.reshape(3, -1)), extra_length=3)
    np.testing.assert_allclose(b.read(actual).reshape(4, 6), expected, atol=1e-6)
