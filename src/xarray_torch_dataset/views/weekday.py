"""Views that select by weekday."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr

from xarray_torch_dataset.base import Indexers, Point, View


class SameWeekdayView(View):
    """The same weekday in each of the `weeks` weeks before the point.

    Selects along ``time``, oldest week first. Unlike `OffsetView` it counts
    durations, not steps, so it also works on a time axis with gaps.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        variables: Sequence[str] | None = None,
        weeks: int = 1,
        missing: Literal["drop", "fill"] = "drop",
    ):
        super().__init__(ds, variables, missing)
        self.weeks = weeks

    def select(self, point: Point) -> Indexers:
        back = np.arange(self.weeks, 0, -1) * np.timedelta64(7, "D")
        return {"time": self.locate("time", point["time"] - back)}
