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
        for dim in ("init_time", "lead_time"):
            if not self.ds.indexes[dim].is_unique:
                raise ValueError(f"Source has repeated labels along {dim!r}.")
        # a limit of None means unbounded on that side
        self.init_time_offset_limits = init_time_offset_limits or (None, None)

        # Sorted copies of both coordinates to search in, and the position
        # in the source that each sorted entry came from.
        init_times, lead_times = self.ds.init_time.values, self.ds.lead_time.values
        self._init_order = np.argsort(init_times)
        self._init_times = init_times[self._init_order]
        self._lead_order = np.argsort(lead_times)
        self._lead_times = lead_times[self._lead_order]

    def select(self, point: Point) -> Indexers:
        init_time = point["init_time"]
        valid_time = init_time + point["lead_time"]

        # An init time can only serve the point if one of the source's lead
        # times reaches the valid time from it...
        earliest = valid_time - self._lead_times[-1]
        latest = valid_time - self._lead_times[0]
        # ...and if it lies in the window around the point's init time.
        lower, upper = self.init_time_offset_limits
        if lower is not None:
            earliest = max(earliest, init_time + lower)
        if upper is not None:
            latest = min(latest, init_time + upper)
        start = np.searchsorted(self._init_times, earliest, side="left")
        stop = np.searchsorted(self._init_times, latest, side="right")

        # The lead time each of these candidates needs to reach the valid
        # time, and where it would be among the source's lead times.
        needed = valid_time - self._init_times[start:stop]
        at = np.searchsorted(self._lead_times, needed)
        at = np.minimum(at, len(self._lead_times) - 1)
        found = np.flatnonzero(self._lead_times[at] == needed)
        if not found.size:
            # None marks a missing position, so the point is rejected or
            # filled with NaN
            return {"init_time": None, "lead_time": None}

        # the candidates are in order of time, so the last match is the latest
        k = found[-1]
        return {
            "init_time": self._init_order[start + k],
            "lead_time": self._lead_order[at[k]],
        }
