import numpy as np
import pytest
import torch
import xarray as xr

from xarray_torch_dataset import OffsetView, PointGrid, ToTensors, XrDataset


def history(source, **kwargs):
    """The point's step and the two before it."""
    return OffsetView(source, ["a"], offsets={"time": [-2, -1, 0]}, **kwargs)


def test_points_a_view_cannot_serve_are_dropped(source, times):
    dataset = XrDataset(PointGrid(time=times), {"x": history(source)})

    assert len(dataset) == len(times) - 2
    # index 0 is the first accepted point, not the first point
    xr.testing.assert_identical(dataset[0]["x"], source[["a"]].isel(time=[0, 1, 2]))


def test_a_point_must_be_accepted_by_every_view(source, times):
    views = {
        "x": history(source),
        "y": OffsetView(source, ["a"], offsets={"time": [1]}),
    }
    dataset = XrDataset(PointGrid(time=times), views)

    assert dataset.indices == list(range(2, 30))
    assert set(dataset[0]) == {"x", "y"}


def test_rejected_points_can_be_reported_instead(source, times):
    with pytest.raises(ValueError, match="2 of 31"):
        XrDataset(PointGrid(time=times), {"x": history(source)}, rejected="raise")


def test_dataset_without_accepted_points_is_an_error(source):
    points = [{"time": np.datetime64("2021-01-01")}]
    with pytest.raises(ValueError, match="No point"):
        XrDataset(points, {"x": history(source)})


def test_filling_view_never_drops_a_point(source, times):
    views = {"x": history(source, unavailable="fill")}
    dataset = XrDataset(PointGrid(time=times), views, rejected="raise")

    assert len(dataset) == len(times)
    assert np.isnan(dataset[0]["x"].a.values[:2]).all()


def test_dropping_view_decides_when_views_are_mixed(source, times):
    views = {
        "x": history(source, unavailable="fill"),
        "y": OffsetView(source, ["a"], offsets={"time": [1]}),
    }
    dataset = XrDataset(PointGrid(time=times), views)

    assert dataset.indices == list(range(30))  # only the last point is dropped


def test_transform_receives_the_sample(source, times):
    dataset = XrDataset(
        PointGrid(time=times),
        {"x": history(source)},
        transform=lambda sample: sorted(sample),
    )
    assert dataset[0] == ["x"]


def test_batches_through_a_dataloader(source, times):
    dataset = XrDataset(
        PointGrid(time=times), {"x": history(source)}, transform=ToTensors()
    )
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=5)))

    assert batch["x"].shape == (5, 1, 3, 4)  # batch, variable, time, lat
    np.testing.assert_array_equal(batch["x"][1, 0], source.a.values[1:4])


def test_a_batch_of_samples_equals_the_single_samples(source, times):
    views = {
        "x": history(source, unavailable="fill"),
        "y": OffsetView(source, ["a"], offsets={"time": [1]}),
    }
    dataset = XrDataset(PointGrid(time=times), views)
    indices = [0, 1, 7, 29, 7]

    batch = dataset.__getitems__(indices)
    assert len(batch) == len(indices)
    for sample, i in zip(batch, indices, strict=True):
        assert set(sample) == {"x", "y"}
        for name in sample:
            xr.testing.assert_identical(sample[name], dataset[i][name])


def test_a_batch_goes_through_the_transform(source, times):
    dataset = XrDataset(
        PointGrid(time=times), {"x": history(source)}, transform=ToTensors()
    )
    batch = dataset.__getitems__([3, 4])
    assert all(
        torch.equal(batch[k]["x"], dataset[i]["x"]) for k, i in enumerate([3, 4])
    )
