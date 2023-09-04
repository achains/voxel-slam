import numpy as np 

from typing import List 
from voxel_slam.types import Point3D


__all__ = ["get_bounding_box", "get_box_centroid", "point_is_in_box"]


def get_bounding_box(voxel_center: Point3D, voxel_size: float) -> List[Point3D]:
    """Get bounding box a voxel

    Parameters
    ----------
    voxel_center: Point3D
        Voxel center point (i.g. centroid)
    voxel_size: float
        Size of voxel edge

    Returns
    -------
    bounding_box: List[Point3D]
        List of voxel bounds (8 points)
    """
    bounds = []
    a = voxel_size / 2
    for x in range(2):
        for y in range(2):
            for z in range(2):
                b_box = (voxel_center[0] + a * (-1 if x == 0 else 1),
                         voxel_center[1] + a * (-1 if y == 0 else 1),
                         voxel_center[2] + a * (-1 if z == 0 else 1))
                bounds.append(b_box)

    return bounds

def get_box_centroid(bounding_box: List[Point3D]) -> Point3D:
    """Get bounding box centroid

    Parameters
    ----------
    bounding_box: List[Point3D]
        List of voxel bounds (8 points)

    Returns
    -------
    centroid: Point3D
        Central point of voxel box
    """
    return np.apply_along_axis(lambda x: (min(x) + max(x)) / 2, 0, bounding_box)

def point_is_in_box(point: Point3D, bounding_box: List[Point3D]) -> bool:
    """Check whether point lies within the voxel bounding box

    Parameters
    ----------
    point: Point3D
        Point to check
    
    Returns
    -------
        is_in_box: bool
            True if point lies within the bounding box, False otherwise 
    """
    bounding_box = np.asarray(bounding_box)
    is_in_box = True 
    for i in range(3):
        is_in_box &= min(bounding_box[:, i]) <= point[i] <= max(bounding_box[:, i]) 

    return is_in_box