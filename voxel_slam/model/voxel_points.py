import open3d as o3d
import numpy as np


__all__ = ["VoxelPoints"]


class VoxelPoints:
    def __init__(self, points: list, pcd_idx: list, color=None) -> None:
        if color is None:
            color = [0.0, 0.0, 0.0]
        self.points = points
        self.pcd_idx = pcd_idx
        self.color = color

    def add_point(self, point, pcd_point_id):
        self.points.append(point)
        self.pcd_idx.append(pcd_point_id)

    def segment_max_plane(self, ransac_distance_threshold):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
        _, inliers = pcd.segment_plane(ransac_distance_threshold, 3, 1000)
        return VoxelPoints(
            list(np.asarray(self.points)[inliers]),
            list(np.asarray(self.pcd_idx)[inliers]),
            color=self.color,
        )

    def get_plane_equation(self):
        c = np.mean(self.points, axis=0)
        A = np.array(self.points) - c
        eigvals, eigvects = np.linalg.eig(A.T @ A)
        min_index = np.argmin(eigvals)
        n = eigvects[:, min_index]

        d = -np.dot(n, c)
        normal = int(np.sign(d)) * n
        d *= np.sign(d)
        return np.asarray([normal[0], normal[1], normal[2], d])
