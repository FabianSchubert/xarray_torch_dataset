"""Ready-made views."""

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr

from xarray_torch_dataset.base import Indexers, Point, View


class OffsetView(View):
    """Variables at fixed positional offsets from the point.

    ``offsets={"time": [-3, -2, -1, 0]}`` selects the point's time step and
    the three steps before it; the offsets need not be contiguous. Dimensions
    without offsets are selected at exactly the point's label and dropped
    from the result. Point entries the source has no index for are ignored.

    Offsets count steps along the source's index, not durations. Steps that
    leave the source are missing: they make the point invalid, or are filled
    with NaN if ``missing="fill"``.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        variables: Sequence[str] | None = None,
        offsets: Mapping[str, Sequence[int]] | None = None,
        missing: Literal["drop", "fill"] = "drop",
    ):
        super().__init__(ds, variables, missing)
        self.offsets = {dim: np.asarray(o) for dim, o in (offsets or {}).items()}
        unindexed = set(self.offsets) - set(self.ds.indexes)
        if unindexed:
            raise ValueError(f"Source has no index for offset dims {unindexed}.")

    def select(self, point: Point) -> Indexers:
        indexers: dict[Hashable, Any] = {}
        for dim, label in point.items():
            if dim not in self.ds.indexes:
                continue
            position = self.locate(dim, label)
            if dim not in self.offsets:
                indexers[dim] = position
                continue
            offsets = self.offsets[dim]
            # a label the source lacks (position -1) has no neighbours either
            indexers[dim] = (
                position + offsets if position >= 0 else np.full_like(offsets, -1)
            )
        absent = set(self.offsets) - set(indexers)
        if absent:
            raise ValueError(f"Point has no label for offset dims {absent}.")
        return indexers
