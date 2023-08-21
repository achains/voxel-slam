from voxel_slam.frontend.filter.base_filter import AbstractVoxelFilter
from sklearn.cluster import AgglomerativeClustering

import numpy as np

__all__ = ["NormalsFilter", "PlaneDistanceFilter", "EmptyVoxelsFilter"]


class NormalsFilter(AbstractVoxelFilter):
    def __init__(self, cosine_distance_threshold) -> None:
        self.cosine_distance_threshold = cosine_distance_threshold

    def filter(self, voxel_feature_map):
        self._filter_normals(voxel_feature_map)
        return super().filter(voxel_feature_map)

    def _filter_normals(self, voxel_feature_map):
        for pose_to_points in voxel_feature_map.values():
            normals = [plane.get_plane_equation()[:-1] for plane in pose_to_points.values()]
            if len(normals) < 2:
                continue
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.cosine_distance_threshold,
                metric="cosine",
                linkage="single",
                compute_distances=True 
            ).fit(np.asarray(normals))

            stable_plane_label = np.bincount(clustering.labels_).argmax()
            outlier_plane_poses = np.asarray(list(pose_to_points.keys()))[clustering.labels_ != stable_plane_label]
            for pose_id in outlier_plane_poses:
                pose_to_points.pop(pose_id)


class PlaneDistanceFilter(AbstractVoxelFilter):
    def filter(self, voxel_feature_map):
        self._filter_plane_distance(voxel_feature_map)
        return super().filter(voxel_feature_map)
    
    def _filter_plane_distance(self, voxel_feature_map):
        for pose_to_points in voxel_feature_map.values():
            planes_d = [plane.get_plane_equation()[-1] for plane in pose_to_points.values()]
            if len(planes_d) < 2:
                continue

            clustering_d = AgglomerativeClustering(
                n_clusters=2,
            ).fit(np.asarray(planes_d).reshape(-1, 1))

            stable_plane_label = np.bincount(clustering_d.labels_).argmax()
            outlier_plane_poses = np.asarray(list(pose_to_points.keys()))[clustering_d.labels_ != stable_plane_label]
            for pose_id in outlier_plane_poses:
                pose_to_points.pop(pose_id)


class EmptyVoxelsFilter(AbstractVoxelFilter):
    def __init__(self, min_voxel_poses=2) -> None:
        self.min_voxel_poses = min_voxel_poses

    def filter(self, voxel_feature_map):
        self._filter_empty_voxels(voxel_feature_map)
        return super().filter(voxel_feature_map)
    
    def _filter_empty_voxels(self, voxel_feature_map):
        empty_voxels = []
        for voxel_center, pose_to_points in  voxel_feature_map.items():
            if len(pose_to_points) < self.min_voxel_poses:
                empty_voxels.append(voxel_center)

        for voxel_center in empty_voxels:
            voxel_feature_map.pop(voxel_center) 