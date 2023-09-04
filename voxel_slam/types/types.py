from nptyping import NDArray, Shape, Float
from typing import Annotated, Literal

__all__ = ["Point3D"]


Point3D = NDArray[Shape["1, 3"], Float]
