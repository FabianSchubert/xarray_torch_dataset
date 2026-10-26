import numpy as np
import pytest
import xarray as xr
from conftest import DAY

from xarray_torch_dataset import OffsetView

OFFSETS = np.array([-3, -1, 0, 2]) * DAY  # not evenly spaced on purpose


def test_selects_labels_relative_to_the_point(source, times):
    view = OffsetView(source, offsets={"time": OFFSETS})
    out = view.fetch({"time": times[10]})
    xr.testing.assert_identical(out, source.sel(time=times[10] + OFFSETS))


def test_dimension_without_offsets_is_selected_exactly_and_dropped(source, times):
    out = OffsetView(source, ["a"]).fetch({"time": times[7]})
    xr.testing.assert_identical(out, source[["a"]].sel(time=times[7]))
    assert "time" not in out.a.dims


def test_point_entries_without_an_index_are_ignored(source, times):
    view = OffsetView(source)
    point = {"time": times[7], "member": 3, "lat": 2}  # lat has no coordinate
    xr.testing.assert_identical(view.fetch(point), source.sel(time=times[7]))


def test_points_whose_window_leaves_the_source_are_rejected(source, times):
    view = OffsetView(source, offsets={"time": OFFSETS})
    accepted = [view.accepts({"time": t}) for t in times]
    assert accepted == [3 <= k <= 28 for k in range(len(times))]


def test_label_missing_from_the_source_is_rejected(source):
    absent = {"time": np.datetime64("2021-06-01")}
    assert not OffsetView(source).accepts(absent)
    assert not OffsetView(source, offsets={"time": [DAY, 2 * DAY]}).accepts(absent)


def test_offsets_are_durations_not_steps(source, times):
    gappy = source.isel(time=[0, 1, 2, 4, 5])  # day 3 is absent
    view = OffsetView(gappy, ["a"], offsets={"time": [-2 * DAY, -DAY, 0 * DAY]})

    # two days back from day 4 is day 2, though it is only one step away
    assert view.select({"time": times[2]})["time"].tolist() == [0, 1, 2]
    # a window that needs day 3 is not served with a neighbouring day
    assert view.select({"time": times[4]})["time"].tolist() == [2, -1, 3]
    assert not view.accepts({"time": times[4]})


def test_a_coarser_source_only_serves_the_points_on_its_grid(source, times):
    coarse = source.isel(time=slice(None, None, 2))
    view = OffsetView(coarse, offsets={"time": [-2 * DAY, 0 * DAY]})
    accepted = [view.accepts({"time": t}) for t in times[:6]]
    assert accepted == [False, False, True, False, True, False]


def test_fill_blanks_only_the_labels_outside_the_source(source, times):
    view = OffsetView(source, ["a"], offsets={"time": OFFSETS}, unavailable="fill")

    start = view.fetch({"time": times[1]})  # days -2, 0, 1, 3
    assert np.isnan(start.a.values[0]).all()
    np.testing.assert_array_equal(start.a.values[1:], source.a.values[[0, 1, 3]])

    end = view.fetch({"time": times[29]})  # days 26, 28, 29, 31
    assert np.isnan(end.a.values[3]).all()
    np.testing.assert_array_equal(end.a.values[:3], source.a.values[[26, 28, 29]])


def test_fill_for_a_label_missing_from_the_source(source):
    view = OffsetView(source, ["a"], offsets={"time": OFFSETS}, unavailable="fill")
    out = view.fetch({"time": np.datetime64("2021-06-01")})
    assert out.a.shape == (4, 4)
    assert out.a.isnull().all()


def test_labels_are_the_point_plus_the_offsets_also_outside_the_source(source, times):
    view = OffsetView(source, offsets={"time": OFFSETS})
    labels = view.labels({"time": times[1], "member": 3})  # days -2, 0, 1, 3

    assert set(labels) == {"time"}
    np.testing.assert_array_equal(labels["time"], times[1] + OFFSETS)
    assert not view.available({"time": times[1]})


def test_fill_shows_the_labels_that_were_asked_for(source, times):
    view = OffsetView(source, ["a"], offsets={"time": OFFSETS}, unavailable="fill")
    out = view.fetch({"time": times[29]})  # days 26, 28, 29, 31

    np.testing.assert_array_equal(out.time.values, times[29] + OFFSETS)
    assert np.isnan(out.a.values[3]).all()

    absent = np.datetime64("2021-06-01")
    assert (
        OffsetView(source, ["a"], unavailable="fill")
        .fetch({"time": absent})
        .time.values
        == absent
    )


def test_offsets_along_other_labels_are_in_their_units():
    levels = xr.Dataset(
        {"t": (("level",), np.arange(4.0))}, coords={"level": [1000, 850, 500, 250]}
    )
    view = OffsetView(levels, offsets={"level": [0, -350]})
    assert view.select({"level": 850})["level"].tolist() == [1, 2]
    assert view.select({"level": 500})["level"].tolist() == [2, -1]


def test_offsets_along_time_must_be_durations(source):
    with pytest.raises(TypeError, match="durations"):
        OffsetView(source, offsets={"time": [-1, 0]})


def test_offsets_need_an_index(source):
    with pytest.raises(ValueError, match="no index"):
        OffsetView(source, offsets={"lat": [0, 1]})


def test_point_needs_a_label_for_every_offset_dimension(source):
    view = OffsetView(source, offsets={"time": [0 * DAY, DAY]})
    with pytest.raises(ValueError, match="no label"):
        view.select({"lat": 0})
