import numpy as np
import xarray as xr
from conftest import DAY

from xarray_torch_dataset import SameWeekdayView


def test_selects_the_same_weekday_in_earlier_weeks_oldest_first(source, times):
    out = SameWeekdayView(source, ["a"], weeks=2).fetch({"time": times[20]})
    xr.testing.assert_identical(out, source[["a"]].isel(time=[6, 13]))


def test_points_without_enough_earlier_weeks_are_invalid(source, times):
    view = SameWeekdayView(source, weeks=2)
    valid = [view.is_valid({"time": t}) for t in times]
    assert valid == [k >= 14 for k in range(len(times))]


def test_counts_days_not_steps(source, times):
    gappy = source.isel(time=[0, 7, 8, 14])  # a time axis with gaps
    out = SameWeekdayView(gappy, ["a"], weeks=2).fetch({"time": times[14]})
    xr.testing.assert_identical(out, source[["a"]].isel(time=[0, 7]))
    assert not SameWeekdayView(gappy, weeks=1).is_valid({"time": times[8]})


def test_fill_blanks_the_weeks_before_the_source(source, times):
    view = SameWeekdayView(source, ["a"], weeks=2, missing="fill")
    out = view.fetch({"time": times[10]})  # 14 days back is before the start

    assert np.isnan(out.a.values[0]).all()
    np.testing.assert_array_equal(out.a.values[1], source.a.values[3])
    assert np.isnat(out.time.values[0])
    assert out.time.values[1] == times[10] - 7 * DAY
