"""Type definitions for synapse."""

from typing import Union, List, Tuple, Literal

DType = Literal["f32", "f64"]
Shape = Union[List[int], Tuple[int, ...]]