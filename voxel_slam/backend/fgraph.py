import mrob 
import numpy as np
import copy 

__all__ = ["FGraph"]


class FGraph:
    def __init__(self, voxel_feature_map, number_of_poses) -> None:
        self.graph = FGraph.init_graph(voxel_feature_map, number_of_poses)
    
    @staticmethod
    def init_graph(voxel_feature_map, number_of_poses):
        graph = mrob.FGraph()

        for i in range(number_of_poses):
            node_mode = mrob.NODE_STANDARD if i > 0 else mrob.NODE_ANCHOR
            pose_id = graph.add_node_pose_3d(mrob.geometry.SE3(), node_mode)
        
        for _ in range(len(voxel_feature_map)):
            feature_id = graph.add_eigen_factor_plane()
        
        for feature_id, voxel_center in enumerate(voxel_feature_map):
            for pose_id, feature_cloud in voxel_feature_map[voxel_center].items():
                graph.eigen_factor_plane_add_points_array(
                    planeEigenId = feature_id,
                    nodePoseId = pose_id,
                    pointsArray = np.asarray(feature_cloud.points),
                    W = 1.0 
                )

        return graph

    def get_optimized_poses(self, number_of_iterations=1000, verbose=True):
        if verbose:
            print("FGraph initial error:", self.graph.chi2(True))

        self.graph.solve(mrob.LM_ELLIPS, number_of_iterations)

        if verbose:
            print("Chi2:", self.graph.chi2())
        
        return self.graph.get_estimated_state()