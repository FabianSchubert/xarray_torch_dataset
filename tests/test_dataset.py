import pickle

import numpy as np
import pytest
import torch
import xarray as xr
from conftest import DAY

from xarray_torch_dataset import OffsetView, PointGrid, ToTensors, XrDataset


def history(source, **kwargs):
    """The point's day and the two before it."""
    return OffsetView(
        source, ["a"], offsets={"time": [-2 * DAY, -DAY, 0 * DAY]}, **kwargs
    )


def test_points_a_view_cannot_serve_are_dropped(source, times):
    dataset = XrDataset(PointGrid(time=times), {"x": history(source)})

    assert len(dataset) == len(times) - 2
    # index 0 is the first accepted point, not the first point
    xr.testing.assert_identical(dataset[0]["x"], source[["a"]].isel(time=[0, 1, 2]))


def test_a_point_must_be_accepted_by_every_view(source, times):
    views = {
        "x": history(source),
        "y": OffsetView(source, ["a"], offsets={"time": [DAY]}),
    }
    dataset = XrDataset(PointGrid(time=times), views)

    assert dataset.indices.tolist() == list(range(2, 30))
    assert set(dataset[0]) == {"x", "y"}


def test_rejected_points_are_counted_for_the_first_view_that_rejects(source, times):
    late = OffsetView(source, ["a"], offsets={"time": [DAY]})  # rejects the last
    views = {"x": history(source), "y": late, "z": history(source)}
    dataset = XrDataset(PointGrid(time=times), views)

    # z rejects the same two points as x, which are counted for x
    assert dataset.n_rejected == {"x": 2, "y": 1, "z": 0}
    assert len(dataset) + sum(dataset.n_rejected.values()) == len(times)


def test_points_can_be_a_plain_list_of_dicts(source, times):
    points = [{"time": t} for t in times[:5]]
    dataset = XrDataset(points, {"x": history(source)})

    assert isinstance(dataset.indices, np.ndarray)
    assert dataset.indices.tolist() == [2, 3, 4]
    xr.testing.assert_identical(dataset[0]["x"], source[["a"]].isel(time=[0, 1, 2]))


def test_rejected_points_can_be_reported_instead(source, times):
    with pytest.raises(ValueError, match="2 of 31 points were rejected: 2 by view 'x'"):
        XrDataset(PointGrid(time=times), {"x": history(source)}, rejected="raise")


def test_dataset_without_accepted_points_is_an_error(source):
    points = [{"time": np.datetime64("2021-01-01")}]
    with pytest.raises(ValueError, match="All 1 points were rejected: 1 by view 'x'"):
        XrDataset(points, {"x": history(source)})


def test_dataset_without_points_is_an_error(source):
    with pytest.raises(ValueError, match="no points"):
        XrDataset([], {"x": history(source)})


def test_without_validation_every_point_is_kept_and_fails_when_fetched(source, times):
    class NeverAsked(OffsetView):
        def accepts(self, point):
            raise AssertionError("accepts must not be called")

    views = {"x": NeverAsked(source, ["a"], offsets={"time": [-2 * DAY, 0 * DAY]})}
    dataset = XrDataset(PointGrid(time=times), views, validate=False)

    assert len(dataset) == len(times)
    assert dataset.n_rejected is None
    xr.testing.assert_identical(dataset[2]["x"], source[["a"]].isel(time=[0, 2]))
    with pytest.raises(IndexError, match="leaves the source"):
        dataset[0]


def test_filling_view_never_drops_a_point(source, times):
    views = {"x": history(source, unavailable="fill")}
    dataset = XrDataset(PointGrid(time=times), views, rejected="raise")

    assert len(dataset) == len(times)
    assert dataset.n_rejected == {"x": 0}
    assert np.isnan(dataset[0]["x"].a.values[:2]).all()


def test_dropping_view_decides_when_views_are_mixed(source, times):
    views = {
        "x": history(source, unavailable="fill"),
        "y": OffsetView(source, ["a"], offsets={"time": [DAY]}),
    }
    dataset = XrDataset(PointGrid(time=times), views)

    assert dataset.indices.tolist() == list(range(30))  # only the last point is dropped


def test_without_stages_the_sample_is_the_fetched_data(source, times):
    dataset = XrDataset(PointGrid(time=times), {"x": history(source)})
    xr.testing.assert_identical(dataset[0]["x"], source[["a"]].isel(time=[0, 1, 2]))


class Names(XrDataset):
    """Prepares the names of the views, and returns them sorted."""

    def prepare(self, data):
        return list(data)

    def transform(self, sample):
        return sorted(sample)


def test_transform_receives_what_prepare_returns(source, times):
    dataset = Names(PointGrid(time=times), {"y": history(source), "x": history(source)})
    assert dataset[0] == ["x", "y"]


class Tensors(XrDataset):
    def transform(self, sample):
        return ToTensors()(sample)


def test_batches_through_a_dataloader(source, times):
    dataset = Tensors(PointGrid(time=times), {"x": history(source)})
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=5)))

    assert batch["x"].shape == (5, 1, 3, 4)  # batch, variable, time, lat
    np.testing.assert_array_equal(batch["x"][1, 0], source.a.values[1:4])


def test_labels_and_availability_of_every_sample(source, times):
    views = {
        "x": history(source, unavailable="fill"),
        "y": OffsetView(source, ["a"], offsets={"time": [DAY]}),  # drops the last
    }
    dataset = XrDataset(PointGrid(time=times), views)

    x = dataset.labels("x")
    assert set(x) == {"time"} and x["time"].shape == (30, 3)
    np.testing.assert_array_equal(x["time"][:, 2], times[:30])
    np.testing.assert_array_equal(x["time"][0], times[0] + [-2 * DAY, -DAY, 0 * DAY])
    np.testing.assert_array_equal(dataset.labels("y")["time"][:, 0], times[1:])
    # the history is complete from the third day on
    assert dataset.available("x").tolist() == [False, False] + [True] * 28
    assert dataset.available("y").all()


def test_labels_without_validation_cover_every_point(source, times):
    dataset = XrDataset(PointGrid(time=times), {"x": history(source)}, validate=False)

    assert dataset.labels("x")["time"].shape == (31, 3)
    assert dataset.available("x").sum() == 29


def test_labels_are_collected_without_reading(source, times):
    class Unreadable(OffsetView):
        def fetch(self, point):
            raise AssertionError("fetch must not be called")

    dataset = XrDataset(PointGrid(time=times), {"x": Unreadable(source)})
    np.testing.assert_array_equal(dataset.labels("x")["time"], times)
    assert dataset.available("x").all()


# -- the arguments of the constructor -------------------------------------------


class Days(XrDataset):
    """A subclass with its own arguments, one of them with a default."""

    def __init__(self, source, times, step=1):
        super().__init__(PointGrid(time=times[::step]), {"x": history(source)})


def test_the_arguments_of_a_subclass_are_recorded_with_defaults(source, times):
    by_position = Days(source, times)
    by_name = Days(times=times, source=source, step=2)

    assert list(by_position.arguments) == ["source", "times", "step"]
    assert by_position.arguments["source"] is source
    assert by_position.arguments["step"] == 1
    assert by_name.arguments["step"] == 2
    assert by_name.arguments["times"] is times


def test_a_plain_dataset_records_its_own_arguments(source, times):
    points = PointGrid(time=times)
    dataset = XrDataset(points, {"x": history(source)}, validate=False)

    assert list(dataset.arguments) == ["points", "views", "rejected", "validate"]
    assert dataset.arguments["points"] is points
    assert dataset.arguments["rejected"] == "drop"
    assert dataset.arguments["validate"] is False


def test_the_arguments_survive_pickling(source, times):
    """DataLoader workers receive the dataset pickled."""
    again = pickle.loads(pickle.dumps(Days(source, times, step=3)))
    assert again.arguments["step"] == 3
    np.testing.assert_array_equal(again.arguments["times"], times)
