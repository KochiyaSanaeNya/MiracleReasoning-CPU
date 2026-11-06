"""Miracle's non-Neural models, executed through OpenCL on the CPU."""

from .config import ModelConfig
from .model import MiracleModel
from .opencl import OpenCLBackend, OpenCLError, list_devices

__all__ = ["ModelConfig", "MiracleModel", "OpenCLBackend", "OpenCLError", "list_devices"]
