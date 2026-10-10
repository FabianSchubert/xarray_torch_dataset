"""Views on forecast sources with an init time and a lead time dimension."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr

from xarray_torch_dataset.base import Indexers, Point, View


class FindLatestForecastView(View):
    """The latest forecast for the point's valid time.

    A point has an init time and a lead time, which add up to a valid time.
    Among the source's init times within `init_time_offset_limits` of the
    point's init time (lower and upper offset, ``None`` for no limit), the
    view selects the most recent one that has a lead time reaching the same
    valid time.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        variables: Sequence[str] | None = None,
        init_time_offset_limits: tuple[np.timedelta64 | None, np.timedelta64 | None]
        | None = None,
        unavailable: Literal["reject", "fill"] = "reject",
    ) -> None:
        super().__init__(ds, variables, unavailable)
        self.init_times = self.ds.init_time.values

        # a limit of None means unbounded on that side
        lower, upper = init_time_offset_limits or (None, None)
        self.init_time_offset_limits = (
            np.timedelta64(-1_000_000, "D") if lower is None else lower,
            np.timedelta64(1_000_000, "D") if upper is None else upper,
        )

    def select(self, point: Point) -> Indexers:
        valid_time = point["init_time"] + point["lead_time"]

        # positions of the init times inside the window around the point's
        lower, upper = self.init_time_offset_limits
        offsets = self.init_times - point["init_time"]
        candidates = np.flatnonzero((offsets >= lower) & (offsets <= upper))

        # the lead time each candidate needs to reach the valid time, as a
        # position along lead_time (-1 where the source has no such lead)
        leads = self.locate("lead_time", valid_time - self.init_times[candidates])
        found = leads >= 0
        if not found.any():
            # None marks a missing position, so the point is dropped or
            # filled with NaN
            return {"init_time": None, "lead_time": None}

        candidates, leads = candidates[found], leads[found]
        latest = np.argmax(self.init_times[candidates])
        return {"init_time": candidates[latest], "lead_time": leads[latest]}
