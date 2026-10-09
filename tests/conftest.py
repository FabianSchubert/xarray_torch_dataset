import numpy as np
import pytest
import xarray as xr

from xarray_torch_dataset import View

TIMES = np.arange("2020-01-01", "2020-02-01", dtype="datetime64[D]")  # 31 days
DAY = np.timedelta64(1, "D")


@pytest.fixture
def times():
    return TIMES


@pytest.fixture
def source():
    """31 daily steps on 4 unlabelled lats; values identify their position."""
    n = len(TIMES)
    return xr.Dataset(
        {
            "a": (("time", "lat"), np.arange(n * 4.0).reshape(n, 4)),
            "count": (("time",), np.arange(n)),
        },
        coords={"time": TIMES, "day_of_year": ("time", np.arange(1, n + 1))},
    )


class Positions(View):
    """Returns the positions stored in the point, so tests can pick them."""

    def select(self, point):
        return dict(point["positions"])


def at(**positions):
    """A point for `Positions`, e.g. ``at(time=[0, 1])``."""
    return {"positions": positions}
