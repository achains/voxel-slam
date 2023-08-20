import open3d as o3d
import numpy as np
import copy 
import time

from sklearn.cluster import AgglomerativeClustering

from voxel_slam.model import VoxelGrid
from voxel_slam.model import PCDPlane 
from voxel_slam.utility import generate_unique_colors

__all__ = ['VoxelFeatureMap']


class VoxelFeatureMap:
    unique_colors = generate_unique_colors(300)
    def __init__(self, clouds, poses, voxel_size):
        self._transformed_clouds = [copy.deepcopy(pcd).transform(pose) for pcd, pose in zip(clouds, poses)]
        self._voxel_to_pose_points_map = self.build_voxel_map_(voxel_size=voxel_size)

    @property
    def transformed_clouds(self):
        return copy.deepcopy(self._transformed_clouds)

    @staticmethod
    def find_cloud_bounds(clouds):
        min_bound = np.full(3, 1e9)
        max_bound = np.full(3, -1e9)
        for pcd in clouds:
            min_bound = np.minimum(min_bound, pcd.get_min_bound())
            max_bound = np.maximum(max_bound, pcd.get_max_bound())
        return min_bound, max_bound

    def build_voxel_map_(self, voxel_size):
        voxel_grid = VoxelGrid(*VoxelFeatureMap.find_cloud_bounds(self._transformed_clouds), voxel_size=voxel_size)
        voxel_to_pose_points_map = {}

        for pose_id, pcd in enumerate(self._transformed_clouds):
            for point_id, point in enumerate(np.asarray(pcd.points)):
                if not np.any(point):
                    continue

                voxel_center = voxel_grid.get_voxel_index(point)
                if voxel_center not in voxel_to_pose_points_map:
                    voxel_to_pose_points_map[voxel_center] = {}
                
                voxel_pose_points = voxel_to_pose_points_map[voxel_center].get(pose_id, PCDPlane(points=[], pcd_idx=[]))
                voxel_pose_points.add_point(point, point_id)
                voxel_to_pose_points_map[voxel_center].update({pose_id: voxel_pose_points})
              
        return voxel_to_pose_points_map
    
    def extract_voxel_features(self, ransac_distance_threshold, points_filter_function=lambda feature_points: True):
        voxel_feature_map = {voxel_id: {} for voxel_id in self._voxel_to_pose_points_map.keys()}

        for voxel_id, pose_to_points in self._voxel_to_pose_points_map.items():
            for pose_id, pcd_plane in pose_to_points.items():
                try:
                    max_plane = pcd_plane.segment_max_plane(ransac_distance_threshold)
                except RuntimeError:
                    continue

                if points_filter_function(np.asarray(max_plane.points)):
                    voxel_feature_map[voxel_id][pose_id] = max_plane

        return voxel_feature_map
    
    @staticmethod
    def filter_redundant_voxels(voxel_feature_map, min_valid_poses=2):
        redundant_voxels = []
        for voxel_center, pose_to_points in  voxel_feature_map.items():
            if len(pose_to_points) < min_valid_poses:
                redundant_voxels.append(voxel_center)

        for voxel_center in redundant_voxels:
            voxel_feature_map.pop(voxel_center)

    @staticmethod
    def filter_voxel_features(voxel_feature_map, cosine_distance_threshold=0.2):
        for voxel_id, pose_to_points in voxel_feature_map.items():
            normals = [plane.get_plane_equation()[:-1] for plane in pose_to_points.values()]
            if len(normals) < 2:
                continue
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=cosine_distance_threshold,
                metric="cosine",
                linkage="single",
                compute_distances=True 
            ).fit(np.asarray(normals))

            stable_plane_label = np.bincount(clustering.labels_).argmax()
            outlier_plane_poses = np.asarray(list(pose_to_points.keys()))[clustering.labels_ != stable_plane_label]
            for pose_id in outlier_plane_poses:
                pose_to_points.pop(pose_id)
            
            # TODO: Unite with previous steps
            # Add filter by D
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
    
    def get_colored_feature_clouds(self, voxel_feature_map, color_method="voxel"):
        time_init_start = time.perf_counter_ns()
        allowed_methods = ["pose", "voxel"]
        if color_method not in allowed_methods:
            raise TypeError(f"Color method has to be one of {'|'.join(allowed_methods)}")
        colored_clouds = self.transformed_clouds

        print("Time (init):", (time.perf_counter_ns() - time_init_start) / 1e9)  

        color_to_voxel_center = {}

        for voxel_id, (voxel_center, pose_to_points) in enumerate(voxel_feature_map.items()):
            for pose_id, pcd_plane in pose_to_points.items():
                cloud_colors = np.asarray(colored_clouds[pose_id].colors)
                if color_method == "voxel":
                    cloud_colors[pcd_plane.pcd_idx] = self.unique_colors[voxel_id]
                    color_to_voxel_center[self.unique_colors[voxel_id]] = voxel_center
                elif color_method == "pose":
                    cloud_colors[pcd_plane.pcd_idx] = self.unique_colors[pose_id]
                colored_clouds[pose_id].colors = o3d.utility.Vector3dVector(cloud_colors)
        
        return colored_clouds, color_to_voxel_center
