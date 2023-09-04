import nptyping

from typing import Annotated, Literal

__all__ = ["Point3D"]


Point3D = Annotated[nptyping.NDArray[float], Literal[3]]
