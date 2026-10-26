"""Ready-made collections of points."""

import numpy as np
from numpy.typing import ArrayLike

from xarray_torch_dataset.base import Point


class PointGrid:
    """All combinations of the given coordinate labels, computed lazily.

    ``PointGrid(time=times, lat=lats)`` has ``len(times) * len(lats)`` points
    of the form ``{"time": ..., "lat": ...}``.
    """

    def __init__(self, **coords: ArrayLike):
        self.coords = {dim: np.asarray(labels) for dim, labels in coords.items()}
        self.shape = tuple(len(labels) for labels in self.coords.values())

    def __len__(self) -> int:
        return int(np.prod(self.shape))

    def __getitem__(self, i: int, /) -> Point:
        if not 0 <= i < len(self):
            raise IndexError(i)
        positions = np.unravel_index(i, self.shape)
        return {
            dim: labels[pos]
            for (dim, labels), pos in zip(self.coords.items(), positions, strict=True)
        }


class PointTable:
    """Points as the rows of label arrays of equal length.

    ``PointTable(time=times, lat=lats)`` has ``len(times)`` points, the i-th
    being ``{"time": times[i], "lat": lats[i]}``. Unlike `PointGrid`, it can
    hold any combinations of labels, such as the points of a grid that pass
    a filter, while still storing only the arrays.
    """

    def __init__(self, **columns: ArrayLike):
        self.columns = {dim: np.asarray(labels) for dim, labels in columns.items()}
        lengths = {dim: len(labels) for dim, labels in self.columns.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"The label arrays differ in length: {lengths}.")
        self._len = next(iter(lengths.values()), 0)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int, /) -> Point:
        if not 0 <= i < len(self):
            raise IndexError(i)
        return {dim: labels[i] for dim, labels in self.columns.items()}
