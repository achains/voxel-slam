from __future__ import annotations
from abc import ABC, abstractmethod


class VoxelFilter(ABC):
    @abstractmethod
    def set_next(self, filter: VoxelFilter) -> VoxelFilter:
        pass

    @abstractmethod
    def filter(self, voxel_feature_map):
        pass


class AbstractVoxelFilter(VoxelFilter):
    _next_filter: VoxelFilter = None

    def set_next(self, filter: VoxelFilter) -> VoxelFilter:
        self._next_filter = filter
        return filter

    @abstractmethod
    def filter(self, voxel_feature_map):
        if self._next_filter:
            self._next_filter.filter(voxel_feature_map)

        return None
