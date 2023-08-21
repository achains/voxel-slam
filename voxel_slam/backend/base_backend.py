from abc import ABC, abstractmethod

import mrob
import numpy as np


class BaseBackend(ABC):
    """Base class for optimization backend

    Parameters
    ----------
    voxel_feature_map: dict
        Dictionary where each voxel is associated with pose features dictionary
    number_of_poses: int
        Total number of poses in feature map
    """

    def __init__(self, voxel_feature_map: dict, number_of_poses: int) -> None:
        self._graph = mrob.FGraph()
        self.graph_poses_number = self.init_poses(number_of_poses)

        number_of_features = len(voxel_feature_map)
        self.graph_feature_number = self.init_features(number_of_features)

        self.add_feature_points(voxel_feature_map)

    def init_poses(self, number_of_poses) -> int:
        pose_id = self._graph.add_node_pose_3d(mrob.geometry.SE3(), mrob.NODE_ANCHOR)
        for _ in range(number_of_poses - 1):
            pose_id = self._graph.add_node_pose_3d(
                mrob.geometry.SE3(), mrob.NODE_STANDARD
            )
        return pose_id

    @abstractmethod
    def init_features(self, number_of_features) -> int:
        pass

    def add_feature_points(self, voxel_feature_map):
        for feature_id, voxel_center in enumerate(voxel_feature_map):
            for pose_id, feature_cloud in voxel_feature_map[voxel_center].items():
                self._graph.eigen_factor_plane_add_points_array(
                    planeEigenId=feature_id,
                    nodePoseId=pose_id,
                    pointsArray=np.asarray(feature_cloud.points),
                    W=1.0,
                )

    def get_optimized_poses(self, number_of_iterations: int, verbose: bool = False):
        """Run FGraph optimization

        Parameters
        ----------
        number_of_iterations: int
            Maximum iterations to converge
        verbose: bool
            Print additional information (Initial error, iterations to converge, chi2)

        Returns
        -------
        optimized_poses: List[SE3]
            List of transformation matrices in voxel_feature_map coordinates system
        is_converged: bool
            True, if optimization has converged else False
        chi2: float
            Chi2 criteria
        """
        if verbose:
            print("FGraph initial error:", self._graph.chi2(True))

        converge_iterations = self._graph.solve(mrob.LM_ELLIPS, number_of_iterations)
        chi2 = self._graph.chi2()
        if verbose:
            print("Iteratios to converge:", converge_iterations)
            print("Chi2:", chi2)

        return self._graph.get_estimated_state(), (converge_iterations != 0), chi2
