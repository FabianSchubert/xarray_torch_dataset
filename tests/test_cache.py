import math

import numpy as np
import pytest
import xarray as xr
import zarr
from conftest import DAY

from xarray_torch_dataset import (
    CacheView,
    OffsetView,
    PointGrid,
    PointTable,
    XrDataset,
)
from xarray_torch_dataset.cache import CACHE_ATTRIBUTE

TIMES = np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]")  # 10 days
N_LAT = 300  # a window of three days is 7.2 kB, more than a small variable


@pytest.fixture
def field():
    """10 daily steps on 300 lats; values identify their position."""
    values = np.arange(len(TIMES) * N_LAT, dtype=float).reshape(len(TIMES), N_LAT)
    return xr.Dataset(
        {"a": (("time", "lat"), values, {"units": "mm"})},
        coords={"time": TIMES, "lat": np.linspace(-90, 90, N_LAT)},
    )


def prepare(data):
    """A prepared sample: the window, the last day it has data for, and a flag."""
    window = data["x"]
    complete = bool(window.a.notnull().all())
    return xr.Dataset(
        {
            "x": (("step", "lat"), window.a.values, window.a.attrs),
            "last": window.time.where(window.a.notnull().all("lat")).max().values,
            "complete": complete,
        },
        coords={"lat": window.lat},
    )


def finish(prepared):
    """The sample: the window as an array, and the flag."""
    return prepared["x"].values, bool(prepared["complete"])


def raw_dataset(field, points=None, **kwargs):
    view = OffsetView(
        field, ["a"], offsets={"time": [-2 * DAY, -DAY, 0 * DAY]}, unavailable="fill"
    )
    points = points or PointGrid(time=TIMES)
    return XrDataset(points, {"x": view}, prepare=prepare, **kwargs)


def assert_same_samples(raw, cached):
    """The same prepared samples, coordinates and attributes included."""
    assert len(raw) == len(cached)
    for i in range(len(raw)):
        xr.testing.assert_identical(raw[i], cached[i])


# -- writing and reading back -------------------------------------------------


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_cached_samples_are_the_prepared_samples(
    tmp_path, field, num_workers, batch_size
):
    raw = raw_dataset(field)
    raw.cache(
        tmp_path / "cache.zarr",
        batch_size=batch_size,
        num_workers=num_workers,
        multiprocessing_context="forkserver",
    )
    assert_same_samples(raw, XrDataset.from_cache(tmp_path / "cache.zarr"))


def test_the_transform_serves_raw_and_cached_samples(tmp_path, field):
    raw = raw_dataset(field)
    raw.transform = finish
    raw.cache(tmp_path / "cache.zarr")
    cached = XrDataset.from_cache(tmp_path / "cache.zarr", transform=finish)
    for i in range(len(raw)):
        (a, flag_a), (b, flag_b) = raw[i], cached[i]
        np.testing.assert_array_equal(a, b)
        assert flag_a == flag_b


def test_a_cached_dataset_can_be_cached_again(tmp_path, field):
    raw_dataset(field).cache(tmp_path / "cache.zarr")
    XrDataset.from_cache(tmp_path / "cache.zarr").cache(tmp_path / "copy.zarr")
    assert_same_samples(
        raw_dataset(field), XrDataset.from_cache(tmp_path / "copy.zarr")
    )


def test_from_cache_gives_a_plain_dataset_also_on_a_subclass(tmp_path, field):
    class Daily(XrDataset):
        def __init__(self, source):
            super().__init__(PointGrid(time=TIMES), {"x": OffsetView(source)})

    raw_dataset(field).cache(tmp_path / "cache.zarr")
    assert type(Daily.from_cache(tmp_path / "cache.zarr")) is XrDataset


def test_only_a_dataset_with_prepare_is_cached(tmp_path, field):
    dataset = raw_dataset(field)
    dataset.prepare = None
    with pytest.raises(ValueError, match="Only a dataset with `prepare`"):
        dataset.cache(tmp_path / "cache.zarr")
    assert not (tmp_path / "cache.zarr").exists()


def test_the_row_of_a_point_has_its_labels(tmp_path, field):
    raw_dataset(field).cache(tmp_path / "cache.zarr")
    view = CacheView(tmp_path / "cache.zarr")
    row = view.fetch({"time": TIMES[4]})
    assert row.time.values == TIMES[4]
    assert view.point_dims == ["time"]


def test_points_of_the_cache_are_those_of_the_dataset_in_order(tmp_path, field):
    raw_dataset(field, PointTable(time=TIMES[[7, 2, 5]])).cache(tmp_path / "cache.zarr")
    points = CacheView(tmp_path / "cache.zarr").points
    np.testing.assert_array_equal(points.columns["time"], TIMES[[7, 2, 5]])


def test_part_of_a_cache_is_read_by_giving_its_points(tmp_path, field):
    raw_dataset(field).cache(tmp_path / "cache.zarr")
    part = PointTable(time=TIMES[[8, 3]])
    assert_same_samples(
        raw_dataset(field, part),
        XrDataset.from_cache(tmp_path / "cache.zarr", points=part),
    )


def test_points_are_found_whatever_the_unit_of_their_labels(tmp_path, field):
    raw_dataset(field).cache(tmp_path / "cache.zarr")
    view = CacheView(tmp_path / "cache.zarr")
    hours = TIMES[4].astype("datetime64[h]")
    assert view.select({"time": hours}) == {"sample": 4}


def test_points_the_cache_lacks_are_rejected(tmp_path, field):
    raw_dataset(field, PointTable(time=TIMES[:5])).cache(tmp_path / "c.zarr")
    all_days = PointGrid(time=TIMES)
    dataset = XrDataset.from_cache(tmp_path / "c.zarr", points=all_days)
    assert len(dataset) == 5 and dataset.n_rejected == {"cache": 5}
    with pytest.raises(ValueError, match="5 of 10 points were rejected"):
        XrDataset.from_cache(tmp_path / "c.zarr", points=all_days, rejected="raise")


def test_attrs_are_kept(tmp_path, field):
    attrs = {"settings": {"crop": [0, 10], "name": "test"}}
    raw_dataset(field).cache(tmp_path / "cache.zarr", attrs=attrs)
    assert CacheView(tmp_path / "cache.zarr").attrs == attrs
    # they belong to the store, not to the samples
    assert XrDataset.from_cache(tmp_path / "cache.zarr")[0].attrs == {}


# -- the layout of the store ---------------------------------------------------


@pytest.mark.parametrize("per_shard", [None, 1, 4])
def test_layout_of_a_cache_store(tmp_path, field, per_shard):
    raw = raw_dataset(field)
    raw.cache(tmp_path / "cache.zarr", samples_per_shard=per_shard)
    group = zarr.open_group(str(tmp_path / "cache.zarr"), mode="r")
    n = len(raw)
    k = n if per_shard is None else per_shard  # a small array fits one shard

    expected = {
        # name: shape, chunks, shards, dtype, dimension names
        "x": ((n, 3, N_LAT), (1, 3, N_LAT), (k, 3, N_LAT), "float64", 3),
        "last": ((n,), (n,), None, "datetime64[s]", 1),  # as xarray keeps it
        "complete": ((n,), (n,), None, "bool", 1),
        "time": ((n,), (n,), None, "datetime64[D]", 1),
        "lat": ((N_LAT,), (N_LAT,), None, "float64", 1),
    }
    assert sorted(group.keys()) == sorted(expected)
    for name, (shape, chunks, shards, dtype, ndim) in expected.items():
        array = group[name]
        assert (array.shape, array.chunks, array.shards) == (shape, chunks, shards)
        assert array.dtype == np.dtype(dtype), name
        assert len(array.metadata.dimension_names) == ndim
    assert group["x"].metadata.dimension_names == ("sample", "step", "lat")
    assert group["x"].attrs["units"] == "mm"
    files = [p for p in (tmp_path / "cache.zarr" / "x").rglob("*") if p.is_file()]
    assert len(files) - 1 == math.ceil(n / k)  # besides the description
    assert group.attrs[CACHE_ATTRIBUTE] == {"points": ["time"], "coords": ["lat"]}
    # what has not been written reads as missing
    assert np.isnan(group["x"].fill_value)
    assert np.isnat(group["last"].fill_value)


def test_an_existing_store_is_replaced_only_when_asked(tmp_path, field):
    path = tmp_path / "cache.zarr"
    raw_dataset(field, PointTable(time=TIMES[:2])).cache(path)
    with pytest.raises(FileExistsError):
        raw_dataset(field).cache(path)
    raw_dataset(field).cache(path, overwrite=True)
    assert len(CacheView(path).points) == len(TIMES)


# -- what a prepared sample has to be ------------------------------------------


def test_prepared_samples_must_be_datasets(tmp_path, field):
    dataset = raw_dataset(field)
    dataset.prepare = lambda data: data["x"].a.values
    with pytest.raises(TypeError, match="returns an xr.Dataset"):
        dataset.cache(tmp_path / "cache.zarr")


def test_coordinates_must_be_the_same_for_every_sample(tmp_path, field):
    dataset = raw_dataset(field)
    dataset.prepare = lambda data: prepare(data).assign_coords(
        day=("step", data["x"].time.values)
    )
    with pytest.raises(ValueError, match="has to be a data variable"):
        dataset.cache(tmp_path / "cache.zarr")


def test_variables_must_have_the_same_shape_for_every_sample(tmp_path, field):
    dataset = raw_dataset(field)
    dataset.prepare = lambda data: prepare(data).isel(
        step=slice(None, 1 + int(data["x"].a.notnull().all()))
    )
    with pytest.raises(ValueError, match="'x' of a prepared sample"):
        dataset.cache(tmp_path / "cache.zarr")


def test_points_and_prepared_samples_must_not_share_names(tmp_path, field):
    dataset = raw_dataset(field)
    dataset.prepare = lambda data: prepare(data).assign(time=1.0)
    with pytest.raises(ValueError, match=r"both have \['time'\]"):
        dataset.cache(tmp_path / "cache.zarr")


def test_a_store_that_is_not_a_cache_is_refused(tmp_path, field):
    field.to_zarr(tmp_path / "field.zarr", zarr_format=3, consolidated=False)
    with pytest.raises(ValueError, match="not a cache store"):
        CacheView(tmp_path / "field.zarr")
