from ._synapse import NDArrayPy
from .typing import DType, Shape
from typing import Union, List

class NDArray:
    """N Dimensional Array, implemented in Rust."""
    
    def __init__(self, data: Union[List, NDArrayPy], shape: Shape = None, dtype: DType = "f64"):

        self._inner : NDArrayPy

        if isinstance(data, NDArrayPy):
            # Wrapping an existing Rust NDArray
            self._inner = data
        else:
            # Creating from Python data
            self._inner = NDArrayPy(data, shape, dtype)
    
    def __add__(self, other: "NDArray") -> "NDArray":
        result = self._inner.__add__(other._inner)
        return NDArray(result)
    
    def __mul__(self, other: "NDArray") -> "NDArray":
        result = self._inner.__mul__(other._inner)
        return NDArray(result)
    
    def __truediv__(self, other: "NDArray") -> "NDArray":
        result = self._inner.__truediv__(other._inner)
        return NDArray(result)
    
    def dot(self, other: "NDArray") -> "NDArray":
        result = self._inner.dot(other._inner)
        return NDArray(result)
    
    def sum_axis(self, axis: int) -> "NDArray":
        result = self._inner.sum_axis(axis)
        return NDArray(result)
    
    def sum(self) -> float:
        return self._inner.sum()
    
    def __repr__(self) -> str:
        return self._inner.__repr__()

__all__ = ["NDArray"]
