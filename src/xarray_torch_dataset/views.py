"""Ready-made views."""

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from xarray_torch_dataset.base import Indexers, Point, View


class OffsetView(View):
    """Variables at fixed positional offsets from the point.

    ``offsets={"time": [-3, -2, -1, 0]}`` selects the point's time step and
    the three steps before it; the offsets need not be contiguous. Dimensions
    without offsets are selected at exactly the point's label and dropped
    from the result. Point entries the source has no index for are ignored.

    Offsets count steps along the source's index, not durations.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        variables: Sequence[str] | None = None,
        offsets: Mapping[str, Sequence[int]] | None = None,
    ):
        super().__init__(ds, variables)
        self.offsets = {dim: np.asarray(o) for dim, o in (offsets or {}).items()}
        missing = set(self.offsets) - set(self.ds.indexes)
        if missing:
            raise ValueError(f"Source has no index for offset dims {missing}.")

    def select(self, point: Point) -> Indexers:
        indexers: dict[Hashable, Any] = {}
        for dim, label in point.items():
            if dim not in self.ds.indexes:
                continue
            if dim not in self.offsets:
                indexers[dim] = label
                continue
            index = self.ds.indexes[dim]
            positions = index.get_loc(label) + self.offsets[dim]
            if positions.min() < 0 or positions.max() >= len(index):
                raise IndexError(f"Offsets around {label!r} leave {dim!r}.")
            indexers[dim] = index[positions]
        absent = set(self.offsets) - set(indexers)
        if absent:
            raise ValueError(f"Point has no label for offset dims {absent}.")
        return indexers
