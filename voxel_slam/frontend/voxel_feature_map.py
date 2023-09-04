import open3d as o3d
import numpy as np
import copy

from sklearn.cluster import AgglomerativeClustering

from collections import namedtuple
from typing import List, Tuple, Dict

from voxel_slam.frontend import voxel_utils
from voxel_slam.model import VoxelGrid
from voxel_slam.model import VoxelPoints
from voxel_slam.utility import generate_unique_colors
from voxel_slam.types import Point3D
from voxel_slam.frontend.filter import EmptyVoxelsFilter, NormalsFilter

__all__ = ["VoxelMap", "VoxelKey"]


VoxelKey = namedtuple("VoxelKey", ["centroid", "size"])


class VoxelMap:
    """Voxel Map representation of aggregated point clouds by their poses

    Parameters
    ----------
    clouds: List[o3d.geometry.PointCloud]
        List of initial point clouds
    poses: List[np.ndarray((4, 4))]
        SE(3) transformation matrices from point cloud coordinate system to map coordinates
    voxel_size: float
        Initial (i.g. maximal) possible size of a voxel in map
    
    Attributes
    ----------
    number_of_poses: int 
        Length of poses 
    transformed_clouds: List[o3d.geometry.PointCloud]
        Point clouds with applied poses (i.g. Point clouds in map coordinate system)
    voxel_to_pose_points_map: Dict[Point3D, Dict[int, VoxelPoints]]
        Voxel map representation. For each voxel center there is a "pose_id-to-points" map.
        i.g. For each voxel we have points lying in it according to pose number
    """
    unique_colors = generate_unique_colors(300)

    def __init__(self, clouds: List[o3d.geometry.PointCloud], poses: List, voxel_size: float) -> None:
        self.number_of_poses = len(poses)

        if len(clouds) != self.number_of_poses:
            raise ValueError(f"Length of poses sequence has to be equal to length of clouds ({self.number_of_poses} != {len(clouds)})")

        self._transformed_clouds = [
            copy.deepcopy(pcd).transform(pose) for pcd, pose in zip(clouds, poses)
        ]
        self._voxel_to_pose_points_map = self._build_voxel_map(voxel_size=voxel_size)

    @property
    def transformed_clouds(self) -> List[o3d.geometry.PointCloud]:
        """Get list of points clouds in map coordinate system according to poses

        Returns
        -------
        transformed_clouds: List[o3d.geometry.PointCloud]
            Point clouds with applied poses
        """
        return copy.deepcopy(self._transformed_clouds)
    
    @staticmethod
    def find_clouds_bounds(clouds: List[o3d.geometry.PointCloud]) -> Tuple[Point3D, Point3D]:
        """Compute minimal and maximal bounds for point cloud sequence

        Parameters
        ----------
        clouds: List[o3d.geometry.PointCloud]
            Point cloud sequence to find bound
            (Note: Ensure that point clouds lie in the same coordinate system)

        Returns 
        -------
        map_bounds: Tuple[Point3D, Point3D]
            Minimal and maximal point of bounding box for list of clouds
        """
        min_bound = np.full(3, 1e9)
        max_bound = np.full(3, -1e9)
        for pcd in clouds:
            min_bound = np.minimum(min_bound, pcd.get_min_bound())
            max_bound = np.maximum(max_bound, pcd.get_max_bound())
        return min_bound, max_bound
    
    def adaptive_feature_extraction(self, ransac_distance_threshold: float, adaptive_voxel_size: float) -> Dict[Point3D, Dict[int, VoxelPoints]]:
        """Plane feature extraction with adaptive voxelization 
        Important note: Adaptive feature extraction changes inner representation of voxel map

        Parameters
        ----------
        ransac_distance_threshold: float
            Maximum distance a point can have to an estimated plane to be considered an inlier
        adaptive_voxel_size: float
            Break down missmatched voxels into octants until their size is greater than adaptive_voxel_size

        Returns
        -------
        voxel_feature_map: Dict[Point3D, Dict[int, VoxelPoints]]
            Voxel map features. For each voxel center there is a "pose_id-to-feature" map.
            i.g. For each voxel we have feature points lying in it according to pose number
        """
        has_breakable_voxels = True
        while has_breakable_voxels:
            voxel_feature_map = self.extract_voxel_features(ransac_distance_threshold)
            EmptyVoxelsFilter(min_voxel_poses=self.number_of_poses).filter(voxel_feature_map)
            inconsistent_voxels = self._find_inconsistent_voxels(voxel_feature_map)
            has_breakable_voxels = self._break_map_on_octants(inconsistent_voxels, adaptive_voxel_size)

        return voxel_feature_map
            
    
    def extract_voxel_features(self, ransac_distance_threshold: float) -> Dict[Point3D, Dict[int, VoxelPoints]]:
        """Plane feature extraction from voxels 

        Parameters
        ----------
        ransac_distance_threshold: float
            Maximum distance a point can have to an estimated plane to be considered an inlier
        adaptive_voxel_size: float, default=None
            Break down missmatched voxels into octants until their size is greater than adaptive_voxel_size
            If None was given skip break down stage

        Returns
        -------
        voxel_feature_map: Dict[Point3D, Dict[int, VoxelPoints]]
            Voxel map features. For each voxel center there is a "pose_id-to-feature" map.
            i.g. For each voxel we have feature points lying in it according to pose number
        """
        voxel_feature_map = {
            voxel_key: {} for voxel_key in self._voxel_to_pose_points_map.keys()
        }
        for voxel_key, pose_to_points in self._voxel_to_pose_points_map.items():
            for pose_id, points in pose_to_points.items():
                try:
                    max_plane = points.segment_max_plane(ransac_distance_threshold)
                except RuntimeError:
                    pass

                voxel_feature_map[voxel_key][pose_id] = max_plane

        return voxel_feature_map
    
    def _break_map_on_octants(self, inconsistent_voxels: List[VoxelKey], adaptive_voxel_size: float) -> bool:
        """Break ambiguous voxels in map into octants.
        Voxel is considered ambiguous if it has normals in different directions throughout the poses.

        Parameters
        ----------
        inconsistent_voxels: List[VoxelKey]
            Voxel keys to break into octants
        adaptive_voxel_size: float
            Break down missmatched voxels into octants until their size is greater than adaptive_voxel_size

        Returns
        -------
        found_inconsistent: bool
            True if we've performed at least one voxel break, False otherwise
        """
        found_inconsistent = False 
        for voxel_key in inconsistent_voxels:
            if voxel_key.size <= adaptive_voxel_size:
                continue

            found_inconsistent = True 
            octant_size = voxel_key.size / 2
            octant_centroids = voxel_utils.get_bounding_box(
                voxel_center=voxel_key.centroid, 
                voxel_size=octant_size
            )
            # Add octant to voxel map
            for oct_center in octant_centroids:
                octant_key = VoxelKey(oct_center, octant_size)
                if octant_key not in self._voxel_to_pose_points_map:
                    self._voxel_to_pose_points_map[octant_key] = {}

            # Assign points lying in octants
            for pose_id, voxel_points in self._voxel_to_pose_points_map[voxel_key].items():
                for point, point_id in zip(voxel_points.points, voxel_points.pcd_idx):
                    # Find point's octant
                    for oct_center in octant_centroids:
                        if voxel_utils.point_is_in_box(point, voxel_utils.get_bounding_box(oct_center, octant_size)):
                            octant_key = VoxelKey(oct_center, octant_size)
                            octo_points = self._voxel_to_pose_points_map[octant_key].get(pose_id, VoxelPoints([], []))
                            octo_points.add_point(point, point_id)
                            self._voxel_to_pose_points_map[octant_key].update({pose_id: octo_points})
            # Pop old voxel center
            self._voxel_to_pose_points_map.pop(voxel_key)

        return found_inconsistent
        
        
    def _find_inconsistent_voxels(self, voxel_feature_map: Dict[Point3D, Dict[int, VoxelPoints]]) -> List[VoxelKey]:
        """Get list of ambiguous inconsistent voxels.
        Voxel is considered ambiguous if it has normals in different directions throughout the poses.

        Parameters
        ----------
        voxel_feature_map: Dict[Point3D, Dict[int, VoxelPoints]]
            Voxel map features. For each voxel center there is a "pose_id-to-feature" map.
            i.g. For each voxel we have feature points lying in it according to pose number

        Returns
        -------
        inconsistent_voxels: List[VoxelKey]
            Keys of ambiguous voxels
        """
        inconsistent_voxels = []

        for voxel_id, pose_to_points in voxel_feature_map.items():
            normals = []

            for pose_id, feature_points in pose_to_points.items():
                normals.append(feature_points.get_plane_equation()[:-1])

            normals_cosine_distance_threshold = 0.2            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=normals_cosine_distance_threshold,
                metric="cosine",
                linkage="single",
                compute_distances=True
            ).fit(np.asarray(normals))

            if clustering.n_clusters_ > 1:
                inconsistent_voxels.append(voxel_id)
        
        return inconsistent_voxels
    
    def _build_voxel_map(self, voxel_size: float) -> Dict[Point3D, Dict[int, VoxelPoints]]:
        """Index points and their poses by owning voxels

        Parameters
        ----------
        voxel_size: float
            Initial (i.g. maximal) possible size of a voxel in map

        Returns 
        -------
        voxel_to_pose_points_map: Dict[Point3D, Dict[int, VoxelPoints]]
            Voxel map representation. For each voxel center there is a "pose_id-to-points" map.
        """
        voxel_grid = VoxelGrid(
            *VoxelMap.find_clouds_bounds(self._transformed_clouds),
            voxel_size=voxel_size,
        )
        voxel_to_pose_points_map = {}

        for pose_id, pcd in enumerate(self._transformed_clouds):
            for point_id, point in enumerate(np.asarray(pcd.points)):
                # Check if point is valid (else they will match with wrong voxel) 
                if not np.any(point):
                    continue

                voxel_center = voxel_grid.get_voxel_coordinates(point)
                voxel_key = VoxelKey(voxel_center, voxel_size)
                if voxel_key not in voxel_to_pose_points_map:
                    voxel_to_pose_points_map[voxel_key] = {}

                voxel_pose_points = voxel_to_pose_points_map[voxel_key].get(
                    pose_id, VoxelPoints(points=[], pcd_idx=[])
                )
                voxel_pose_points.add_point(point, point_id)
                voxel_to_pose_points_map[voxel_key].update({pose_id: voxel_pose_points})
        
        return voxel_to_pose_points_map
