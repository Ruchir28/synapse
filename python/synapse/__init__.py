
"""Synapse: A fast n-dimensional array library for Python implemented in Rust."""

from ._core import NDArray
from .typing import DType, Shape

__version__ = "0.1.0"
__all__ = ["NDArray", "DType", "Shape"]