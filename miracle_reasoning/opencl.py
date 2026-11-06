"""Small OpenCL 1.2 binding. Only NumPy and an installed CPU ICD are needed."""
from __future__ import annotations

import ctypes as C
import ctypes.util
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class OpenCLError(RuntimeError):
    pass


def _check(code, operation):
    if code:
        raise OpenCLError(f"{operation} failed (OpenCL error {code})")


class _API:
    def __init__(self):
        name = os.environ.get("MIRACLE_OPENCL_LIBRARY")
        name = name or ("OpenCL.dll" if os.name == "nt" else C.util.find_library("OpenCL"))
        if not name:
            raise OpenCLError("OpenCL loader not found. Install a CPU OpenCL ICD (PoCL or Intel).")
        try:
            self.lib = (C.WinDLL if os.name == "nt" else C.CDLL)(name)
        except OSError as exc:
            raise OpenCLError(f"Cannot load OpenCL library {name}: {exc}") from exc
        ptr, uint, size, integer, bits = C.c_void_p, C.c_uint, C.c_size_t, C.c_int, C.c_uint64
        signatures = {
            "clGetPlatformIDs": (integer, [uint, ptr, ptr]),
            "clGetPlatformInfo": (integer, [ptr, uint, size, ptr, ptr]),
            "clGetDeviceIDs": (integer, [ptr, bits, uint, ptr, ptr]),
            "clGetDeviceInfo": (integer, [ptr, uint, size, ptr, ptr]),
            "clCreateContext": (ptr, [ptr, uint, ptr, ptr, ptr, ptr]),
            "clCreateCommandQueue": (ptr, [ptr, ptr, bits, ptr]),
            "clCreateProgramWithSource": (ptr, [ptr, uint, ptr, ptr, ptr]),
            "clBuildProgram": (integer, [ptr, uint, ptr, C.c_char_p, ptr, ptr]),
            "clGetProgramBuildInfo": (integer, [ptr, ptr, uint, size, ptr, ptr]),
            "clCreateKernel": (ptr, [ptr, C.c_char_p, ptr]),
            "clCreateBuffer": (ptr, [ptr, bits, size, ptr, ptr]),
            "clSetKernelArg": (integer, [ptr, uint, size, ptr]),
            "clEnqueueNDRangeKernel": (integer, [ptr, ptr, uint, ptr, ptr, ptr, uint, ptr, ptr]),
            "clEnqueueReadBuffer": (integer, [ptr, ptr, uint, size, size, ptr, uint, ptr, ptr]),
            "clEnqueueWriteBuffer": (integer, [ptr, ptr, uint, size, size, ptr, uint, ptr, ptr]),
            "clEnqueueCopyBuffer": (integer, [ptr, ptr, ptr, size, size, size, uint, ptr, ptr]),
            "clFinish": (integer, [ptr]),
        }
        for kind in ("Context", "CommandQueue", "Program", "Kernel", "MemObject"):
            signatures[f"clRelease{kind}"] = (integer, [ptr])
        for name, (result, args) in signatures.items():
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = result, args
            setattr(self, name, fn)

    def info(self, fn, handle, key, dtype=None):
        length = C.c_size_t()
        _check(fn(handle, key, 0, None, C.byref(length)), "query info size")
        buf = C.create_string_buffer(length.value)
        _check(fn(handle, key, length, buf, None), "query info")
        return dtype.from_buffer_copy(buf).value if dtype else buf.value.decode("utf-8", "replace")


@dataclass(frozen=True)
class Device:
    platform_index: int
    device_index: int
    platform: str
    name: str
    vendor: str
    kind: str
    memory_bytes: int
    max_allocation_bytes: int
    handle: int

    def to_dict(self):
        return {key: value for key, value in vars(self).items() if key != "handle"}


def _devices(api):
    count = C.c_uint()
    status = api.clGetPlatformIDs(0, None, C.byref(count))
    if status == -1001:  # CL_PLATFORM_NOT_FOUND_KHR
        return []
    _check(status, "enumerate OpenCL platforms")
    platforms = (C.c_void_p * count.value)()
    _check(api.clGetPlatformIDs(count, platforms, None), "enumerate OpenCL platforms")
    result = []
    for pi, platform in enumerate(platforms):
        number = C.c_uint()
        code = api.clGetDeviceIDs(platform, 0xFFFFFFFF, 0, None, C.byref(number))
        if code == -1:  # CL_DEVICE_NOT_FOUND
            continue
        _check(code, "enumerate OpenCL devices")
        handles = (C.c_void_p * number.value)()
        _check(api.clGetDeviceIDs(platform, 0xFFFFFFFF, number, handles, None), "enumerate OpenCL devices")
        for di, handle in enumerate(handles):
            info = lambda key, dtype=None: api.info(api.clGetDeviceInfo, handle, key, dtype)
            kind = info(0x1000, C.c_uint64)
            if not info(0x1027, C.c_uint) or not info(0x1028, C.c_uint):
                continue  # unavailable device or no source compiler
            result.append(Device(pi, di, api.info(api.clGetPlatformInfo, platform, 0x0902),
                                 info(0x102B), info(0x102C),
                                 "cpu" if kind & 2 else "gpu" if kind & 4 else "other",
                                 info(0x101F, C.c_uint64), info(0x1010, C.c_uint64), handle))
    return result


def list_devices():
    """List available devices with a compiler, including GPUs for diagnostics."""
    return _devices(_API())


class Tensor:
    def __init__(self, backend, handle, shape):
        self.backend, self.handle, self.shape = backend, handle, tuple(shape)
        self.size = math.prod(self.shape)

    def __del__(self):
        if getattr(self, "handle", None):
            self.backend.api.clReleaseMemObject(self.handle)
            self.handle = None


class OpenCLBackend:
    """FP32 execution on a CPU device. GPU execution requires explicit opt-in.

    An instance and its command queue must be used by a single thread at a time.
    Never enable fast-relaxed-math: normalization and attention need finite math.
    """

    def __init__(self, *, platform_index=None, device_index=None, device_type="cpu"):
        if device_type not in ("cpu", "gpu"):
            raise ValueError("device_type must be cpu or gpu")
        self.api = _API()
        candidates = [d for d in _devices(self.api) if d.kind == device_type
                      and (platform_index is None or d.platform_index == platform_index)
                      and (device_index is None or d.device_index == device_index)]
        if not candidates:
            raise OpenCLError(f"No matching OpenCL {device_type.upper()} device with a compiler. "
                              "Install a CPU ICD such as PoCL or Intel CPU Runtime; "
                              "use 'miracle-reasoning devices' to inspect devices. No automatic GPU fallback.")
        self.device = candidates[0]
        self.context = self.queue = self.program = None
        self.kernels = {}
        err = C.c_int()
        devices = (C.c_void_p * 1)(self.device.handle)
        self.context = self.api.clCreateContext(None, 1, devices, None, None, C.byref(err))
        _check(err.value, "create context")
        self.queue = self.api.clCreateCommandQueue(self.context, self.device.handle, 0, C.byref(err))
        _check(err.value, "create queue")
        source = (Path(__file__).with_name("kernels.cl")).read_bytes()
        sources = (C.c_char_p * 1)(source)
        self.program = self.api.clCreateProgramWithSource(self.context, 1, sources, None, C.byref(err))
        _check(err.value, "create program")
        code = self.api.clBuildProgram(self.program, 1, devices, b"-cl-std=CL1.2", None, None)
        if code:
            size = C.c_size_t()
            self.api.clGetProgramBuildInfo(self.program, self.device.handle, 0x1183, 0, None, C.byref(size))
            log = C.create_string_buffer(size.value)
            self.api.clGetProgramBuildInfo(self.program, self.device.handle, 0x1183, size, log, None)
            raise OpenCLError(f"OpenCL kernel build failed ({code}):\n{log.value.decode('utf-8', 'replace')}")

    def __del__(self):
        if not hasattr(self, "api"):
            return
        for handle in getattr(self, "kernels", {}).values():
            self.api.clReleaseKernel(handle)
        for name, release in (("queue", "clReleaseCommandQueue"), ("program", "clReleaseProgram"),
                              ("context", "clReleaseContext")):
            if getattr(self, name, None):
                getattr(self.api, release)(getattr(self, name))

    def empty(self, shape):
        shape = tuple(shape)
        if not shape or any(not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError(f"invalid tensor shape: {shape}")
        size = math.prod(shape) * 4
        if size > self.device.max_allocation_bytes:
            raise OpenCLError(f"Buffer needs {size:,} bytes, above device allocation limit "
                              f"{self.device.max_allocation_bytes:,}; reduce context or weight tile size")
        err = C.c_int()
        handle = self.api.clCreateBuffer(self.context, 1, size, None, C.byref(err))
        _check(err.value, f"allocate {size:,} bytes")
        return Tensor(self, handle, shape)

    def upload(self, array):
        array = np.ascontiguousarray(array, dtype=np.float32)
        result = self.empty(array.shape)
        _check(self.api.clEnqueueWriteBuffer(self.queue, result.handle, 1, 0, array.nbytes,
                                            array.ctypes.data, 0, None, None), "upload tensor")
        return result

    def read(self, tensor):
        self._owns(tensor)
        array = np.empty(tensor.shape, dtype=np.float32)
        _check(self.api.clEnqueueReadBuffer(self.queue, tensor.handle, 1, 0, array.nbytes,
                                           array.ctypes.data, 0, None, None), "read tensor")
        return array

    def _owns(self, *tensors):
        if any(t.backend is not self for t in tensors):
            raise ValueError("Tensors belong to different OpenCL contexts")

    def copy(self, source, target, *, source_offset=0, target_offset=0, count=None):
        self._owns(source, target)
        count = source.size - source_offset if count is None else count
        if min(source_offset, target_offset, count) < 0 or source_offset+count > source.size or target_offset+count > target.size:
            raise ValueError("tensor copy is out of bounds")
        if count:
            _check(self.api.clEnqueueCopyBuffer(self.queue, source.handle, target.handle,
                                               source_offset*4, target_offset*4, count*4,
                                               0, None, None), "copy tensor")

    def run(self, name, shape, *args):
        if name not in self.kernels:
            err = C.c_int()
            handle = self.api.clCreateKernel(self.program, name.encode(), C.byref(err))
            _check(err.value, f"create kernel {name}")
            self.kernels[name] = handle
        kernel = self.kernels[name]
        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                self._owns(arg)
                value = C.c_void_p(arg.handle)
            elif isinstance(arg, (float, np.floating)):
                value = C.c_float(arg)
            elif isinstance(arg, (int, np.integer)) and -(2**31) <= arg < 2**31:
                value = C.c_int(arg)
            else:
                raise ValueError(f"unsupported kernel argument: {arg}")
            _check(self.api.clSetKernelArg(kernel, i, C.sizeof(value), C.byref(value)), f"set {name} arg {i}")
        sizes = (C.c_size_t * len(shape))(*shape)
        _check(self.api.clEnqueueNDRangeKernel(self.queue, kernel, len(shape), None, sizes,
                                              None, 0, None, None), f"enqueue {name}")

    def linear(self, x, weight):
        if len(x.shape) != 2 or len(weight.shape) != 2 or x.shape[1] != weight.shape[1]:
            raise ValueError("linear needs x[rows,in] and weight[out,in]")
        rows, inner = x.shape
        out = self.empty((rows, weight.shape[0]))
        self.run("linear", (weight.shape[0], rows), x, weight, out, rows, weight.shape[0], inner)
        return out

    def norm(self, x, weight, width, eps):
        if x.size % width or (weight is not None and weight.shape != (width,)):
            raise ValueError("RMSNorm shape mismatch")
        out = self.empty(x.shape)
        self.run("rms", (x.size//width,), x, weight if weight is not None else x, out,
                 x.size//width, width, float(eps), int(weight is not None))
        return out

    def rotate(self, x, frequencies, positions, heads, dim, split_half=False):
        if x.shape != (positions.size, heads*dim):
            raise ValueError("RoPE shape mismatch")
        out = self.empty(x.shape)
        self.run("rope", (x.size//2,), x, frequencies, positions, out, positions.size,
                 heads, dim, int(split_half))
        return out

    def add(self, x, y, scale=1.0):
        if x.shape != y.shape:
            raise ValueError("residual shapes differ")
        out = self.empty(x.shape)
        self.run("axpy", (x.size,), x, y, out, x.size, float(scale))
        return out

    def swiglu(self, gate, up):
        if gate.shape != up.shape:
            raise ValueError("SwiGLU shapes differ")
        out = self.empty(gate.shape)
        self.run("swiglu", (gate.size,), gate, up, out, gate.size)
        return out

    def softmax(self, x):
        out = self.empty(x.shape)
        self.run("softmax", (x.shape[0],), x, out, *x.shape)
        return out

    def attention(self, query, keys, values, *, heads, kv_heads, dim, start, length,
                  extra_keys=None, extra_values=None, extra_length=0):
        scores = self.empty((heads, length+extra_length))
        self.run("scores", (length+extra_length, heads), query, keys,
                 extra_keys if extra_keys is not None else keys, scores,
                 heads, kv_heads, dim, keys.shape[0], start, length, extra_length)
        probs = self.softmax(scores)
        return self.mix(probs, values, heads, kv_heads, dim, start, length,
                        extra_values, extra_length)

    def mix(self, probs, values, heads, kv_heads, dim, start, length,
            extra_values=None, extra_length=0):
        out = self.empty((1, heads*dim))
        self.run("mix", (heads*dim,), probs, values,
                 extra_values if extra_values is not None else values, out, heads,
                 kv_heads, dim, values.shape[0], start, length, extra_length)
        return out

    def finish(self):
        _check(self.api.clFinish(self.queue), "finish queue")
