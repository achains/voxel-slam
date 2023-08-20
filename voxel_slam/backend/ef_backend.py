from voxel_slam.backend.base_backend import BaseBackend

__all__ = ["EFBackend"]


class EFBackend(BaseBackend):
    def init_features(self, number_of_features) -> int:
        feature_id = 0
        for _ in range(number_of_features):
            feature_id = self._graph.add_eigen_factor_plane()
        return feature_id
