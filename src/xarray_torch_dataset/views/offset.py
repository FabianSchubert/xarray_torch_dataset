"""Selections at fixed offsets from a point's labels."""

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr

from xarray_torch_dataset.base import Indexers, Point, View


class OffsetView(View):
    """Variables at fixed offsets from the point's labels.

    ``offsets={"time": np.arange(-3, 1) * np.timedelta64(1, "h")}`` selects
    the point's time and the three hours before it; the offsets need not be
    evenly spaced. Dimensions without offsets are selected at exactly the
    point's label and dropped from the result. Point entries the source has
    no index for are ignored.

    Offsets are added to the point's label and the result is looked up in
    the source, so along time they are durations, not numbers of steps. A
    label the source lacks is unavailable, also inside a window: the view
    rejects the point, or fills that entry with NaN if
    ``unavailable="fill"``.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        variables: Sequence[str] | None = None,
        offsets: Mapping[str, Any] | None = None,
        unavailable: Literal["reject", "fill"] = "reject",
    ):
        super().__init__(ds, variables, unavailable)
        self.offsets = {dim: np.asarray(o) for dim, o in (offsets or {}).items()}
        unindexed = set(self.offsets) - set(self.ds.indexes)
        if unindexed:
            raise ValueError(f"Source has no index for offset dims {unindexed}.")
        for dim, offset in self.offsets.items():
            # an integer added to a time counts in the unit of that time
            # (nanoseconds, typically), which is never what was meant
            is_time = np.issubdtype(self.ds.indexes[dim].dtype, np.datetime64)
            if is_time and offset.dtype.kind != "m":
                raise TypeError(
                    f"Offsets along {dim!r} must be durations, "
                    'e.g. np.arange(-3, 1) * np.timedelta64(1, "h").'
                )

    def select(self, point: Point) -> Indexers:
        indexers: dict[Hashable, Any] = {}
        for dim, label in point.items():
            if dim not in self.ds.indexes:
                continue
            if dim in self.offsets:
                label = label + self.offsets[dim]
            indexers[dim] = self.locate(dim, label)
        absent = set(self.offsets) - set(indexers)
        if absent:
            raise ValueError(f"Point has no label for offset dims {absent}.")
        return indexers
