import numpy as np
import pytest
import xarray as xr
from conftest import DAY

from xarray_torch_dataset import OffsetView

OFFSETS = np.array([-3, -1, 0, 2])  # not contiguous on purpose


def test_selects_steps_relative_to_the_point(source, times):
    view = OffsetView(source, offsets={"time": OFFSETS})
    out = view.fetch({"time": times[10]})
    xr.testing.assert_identical(out, source.sel(time=times[10] + OFFSETS * DAY))


def test_dimension_without_offsets_is_selected_exactly_and_dropped(source, times):
    out = OffsetView(source, ["a"]).fetch({"time": times[7]})
    xr.testing.assert_identical(out, source[["a"]].sel(time=times[7]))
    assert "time" not in out.a.dims


def test_point_entries_without_an_index_are_ignored(source, times):
    view = OffsetView(source)
    point = {"time": times[7], "member": 3, "lat": 2}  # lat has no coordinate
    xr.testing.assert_identical(view.fetch(point), source.sel(time=times[7]))


def test_points_whose_window_leaves_the_source_are_invalid(source, times):
    view = OffsetView(source, offsets={"time": OFFSETS})
    valid = [view.is_valid({"time": t}) for t in times]
    assert valid == [3 <= k <= 28 for k in range(len(times))]


def test_label_missing_from_the_source_is_invalid(source):
    absent = {"time": np.datetime64("2021-06-01")}
    assert not OffsetView(source).is_valid(absent)
    # offsets around a failed lookup must not land on real positions
    assert not OffsetView(source, offsets={"time": [1, 2]}).is_valid(absent)


def test_fill_blanks_only_the_steps_outside_the_source(source, times):
    view = OffsetView(source, ["a"], offsets={"time": OFFSETS}, missing="fill")

    start = view.fetch({"time": times[1]})  # positions [-2, 0, 1, 3]
    assert np.isnan(start.a.values[0]).all()
    np.testing.assert_array_equal(start.a.values[1:], source.a.values[[0, 1, 3]])

    end = view.fetch({"time": times[29]})  # positions [26, 28, 29, 31]
    assert np.isnan(end.a.values[3]).all()
    np.testing.assert_array_equal(end.a.values[:3], source.a.values[[26, 28, 29]])


def test_fill_for_a_label_missing_from_the_source(source):
    view = OffsetView(source, ["a"], offsets={"time": OFFSETS}, missing="fill")
    out = view.fetch({"time": np.datetime64("2021-06-01")})
    assert out.a.shape == (4, 4)
    assert out.a.isnull().all()


def test_offsets_need_an_index(source):
    with pytest.raises(ValueError, match="no index"):
        OffsetView(source, offsets={"lat": [0, 1]})


def test_point_needs_a_label_for_every_offset_dimension(source):
    view = OffsetView(source, offsets={"time": [0, 1]})
    with pytest.raises(ValueError, match="no label"):
        view.select({"lat": 0})
