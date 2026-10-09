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
