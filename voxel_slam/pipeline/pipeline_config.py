from dataclasses import dataclass

__all__ = ["PipelineConfig"]


@dataclass
class PipelineConfig:
    voxel_size: float 
    ransac_distance_threshold: float
    filter_cosine_distance_threshold: float 
    filter_min_valid_poses: int = 2
    backend_number_of_iterations: int = 1000
    backend_verbose: bool = False
