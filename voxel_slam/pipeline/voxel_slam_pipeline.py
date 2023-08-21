from voxel_slam.frontend.voxel_feature_map import VoxelFeatureMap
from voxel_slam.frontend.filter import VoxelFilter
from voxel_slam.backend.base_backend import BaseBackend
from voxel_slam.pipeline import PipelineConfig

from collections import namedtuple

__all__ = ["VoxelSLAMPipeline", "VoxelSLAMPipelineOutput"]


VoxelSLAMPipelineOutput = namedtuple("VoxelSLAMPipelineOutput", ["optimized_clouds", 
                                                                 "optimized_poses",
                                                                 "is_converged",
                                                                 "chi2",
                                                                 "color_to_voxel_id",
                                                                 ])


class VoxelSLAMPipeline:
    def __init__(self,
                 feature_filter: VoxelFilter,
                 optimization_backend: BaseBackend,
                 config: PipelineConfig) -> None:
        self._feature_filter = feature_filter
        self._optimization_backend = optimization_backend
        self._config = config

    def process(self, clouds, poses) -> VoxelSLAMPipelineOutput:
        voxel_map = VoxelFeatureMap(clouds, poses, voxel_size=self._config.voxel_size)
        feature_map = voxel_map.extract_voxel_features(ransac_distance_threshold=self._config.ransac_distance_threshold)
        self._feature_filter.filter(feature_map)
        backend = self._optimization_backend(feature_map, len(poses))
        optimized_poses, is_converged, chi2 = backend.get_optimized_poses(self._config.backend_number_of_iterations, 
                                                                    self._config.backend_verbose)
        color_to_voxel_id = None
        if self._config.voxel_color_method is not None:
            optimized_clouds, color_to_voxel_id = voxel_map.get_colored_feature_clouds(feature_map, 
                                                                                     color_method=self._config.voxel_color_method)
        else:
            optimized_clouds = voxel_map.transformed_clouds
        
        return VoxelSLAMPipelineOutput(optimized_clouds, optimized_poses, is_converged, chi2, color_to_voxel_id)     
