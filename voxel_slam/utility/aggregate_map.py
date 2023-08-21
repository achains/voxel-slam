import open3d as o3d
import copy

__all__ = ["aggregate_map"]


def aggregate_map(clouds, poses, enable_color=False, enable_uniform_downsample=False):
    cloud_map = o3d.geometry.PointCloud()
    for i in range(len(clouds)):
        pcd = copy.deepcopy(clouds[i])
        if enable_color:
            s = i / len(clouds)
            pcd.paint_uniform_color([0.5, 1 - s, (1 - s) / 2])
        pcd.transform(poses[i])
        cloud_map += pcd.uniform_down_sample(2) if enable_uniform_downsample else pcd

    return cloud_map
