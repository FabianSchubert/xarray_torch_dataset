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
from xarray_torch_dataset.arguments import describe
from xarray_torch_dataset.cache import (
    ARGUMENTS_GROUP,
    CACHE_ATTRIBUTE,
    read_arguments,
)

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


class Windows(XrDataset):
    """Three days up to a point, prepared; the prepared samples are the samples."""

    def prepare(self, data):
        return prepare(data)


class Finished(Windows):
    """`Windows` with `finish` as the cheap stage."""

    def transform(self, sample):
        return finish(sample)


def raw_dataset(field, points=None, cls=Windows, **kwargs):
    view = OffsetView(
        field, ["a"], offsets={"time": [-2 * DAY, -DAY, 0 * DAY]}, unavailable="fill"
    )
    return cls(points or PointGrid(time=TIMES), {"x": view}, **kwargs)


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
    assert_same_samples(raw, Windows.from_cache(tmp_path / "cache.zarr"))


def test_the_transform_serves_raw_and_cached_samples(tmp_path, field):
    raw = raw_dataset(field, cls=Finished)
    raw.cache(tmp_path / "cache.zarr")
    cached = Finished.from_cache(tmp_path / "cache.zarr")
    assert cached.cached and not raw.cached
    for i in range(len(raw)):
        (a, flag_a), (b, flag_b) = raw[i], cached[i]
        np.testing.assert_array_equal(a, b)
        assert flag_a == flag_b


def test_a_cached_dataset_can_be_cached_again(tmp_path, field):
    raw_dataset(field).cache(tmp_path / "cache.zarr")
    Windows.from_cache(tmp_path / "cache.zarr").cache(tmp_path / "copy.zarr")
    assert_same_samples(raw_dataset(field), Windows.from_cache(tmp_path / "copy.zarr"))


def test_from_cache_gives_the_class_without_running_its_init(tmp_path, field):
    class Daily(Windows):
        def __init__(self, source):
            super().__init__(PointGrid(time=TIMES), {"x": OffsetView(source)})
            self.source = source

    raw_dataset(field).cache(tmp_path / "cache.zarr")
    cached = Daily.from_cache(tmp_path / "cache.zarr")
    assert type(cached) is Daily
    assert not hasattr(cached, "source")


def test_only_a_dataset_with_prepare_is_cached(tmp_path, field):
    dataset = raw_dataset(field, cls=XrDataset)
    with pytest.raises(ValueError, match="overrides `prepare`"):
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
        Windows.from_cache(tmp_path / "cache.zarr", points=part),
    )


def test_points_are_found_whatever_the_unit_of_their_labels(tmp_path, field):
    raw_dataset(field).cache(tmp_path / "cache.zarr")
    view = CacheView(tmp_path / "cache.zarr")
    hours = TIMES[4].astype("datetime64[h]")
    assert view.select({"time": hours}) == {"sample": 4}


def test_points_the_cache_lacks_are_rejected(tmp_path, field):
    raw_dataset(field, PointTable(time=TIMES[:5])).cache(tmp_path / "c.zarr")
    all_days = PointGrid(time=TIMES)
    dataset = Windows.from_cache(tmp_path / "c.zarr", points=all_days)
    assert len(dataset) == 5 and dataset.n_rejected == {"cache": 5}
    with pytest.raises(ValueError, match="5 of 10 points were rejected"):
        Windows.from_cache(tmp_path / "c.zarr", points=all_days, rejected="raise")


def test_attrs_are_kept(tmp_path, field):
    attrs = {"settings": {"crop": [0, 10], "name": "test"}}
    raw_dataset(field).cache(tmp_path / "cache.zarr", attrs=attrs)
    assert CacheView(tmp_path / "cache.zarr").attrs == attrs
    # they belong to the store, not to the samples
    assert Windows.from_cache(tmp_path / "cache.zarr")[0].attrs == {}


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
    assert sorted(group.array_keys()) == sorted(expected)
    assert list(group.group_keys()) == [ARGUMENTS_GROUP]
    for name, (shape, chunks, shards, dtype, ndim) in expected.items():
        array = group[name]
        assert (array.shape, array.chunks, array.shards) == (shape, chunks, shards)
        assert array.dtype == np.dtype(dtype), name
        assert len(array.metadata.dimension_names) == ndim
    assert group["x"].metadata.dimension_names == ("sample", "step", "lat")
    assert group["x"].attrs["units"] == "mm"
    files = [p for p in (tmp_path / "cache.zarr" / "x").rglob("*") if p.is_file()]
    assert len(files) - 1 == math.ceil(n / k)  # besides the description
    layout = group.attrs[CACHE_ATTRIBUTE]
    assert (layout["points"], layout["coords"]) == (["time"], ["lat"])
    assert set(layout["arguments"]) == {"points", "views", "rejected", "validate"}
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
    class Values(XrDataset):
        def prepare(self, data):
            return data["x"].a.values

    dataset = raw_dataset(field, cls=Values)
    with pytest.raises(TypeError, match="returns an xr.Dataset"):
        dataset.cache(tmp_path / "cache.zarr")


def test_coordinates_must_be_the_same_for_every_sample(tmp_path, field):
    class Days(XrDataset):
        def prepare(self, data):
            return prepare(data).assign_coords(day=("step", data["x"].time.values))

    dataset = raw_dataset(field, cls=Days)
    with pytest.raises(ValueError, match="has to be a data variable"):
        dataset.cache(tmp_path / "cache.zarr")


def test_variables_must_have_the_same_shape_for_every_sample(tmp_path, field):
    class Shorter(XrDataset):
        def prepare(self, data):
            complete = int(data["x"].a.notnull().all())
            return prepare(data).isel(step=slice(None, 1 + complete))

    dataset = raw_dataset(field, cls=Shorter)
    with pytest.raises(ValueError, match="'x' of a prepared sample"):
        dataset.cache(tmp_path / "cache.zarr")


def test_points_and_prepared_samples_must_not_share_names(tmp_path, field):
    class Clashing(XrDataset):
        def prepare(self, data):
            return prepare(data).assign(time=1.0)

    dataset = raw_dataset(field, cls=Clashing)
    with pytest.raises(ValueError, match=r"both have \['time'\]"):
        dataset.cache(tmp_path / "cache.zarr")


def test_a_store_that_is_not_a_cache_is_refused(tmp_path, field):
    field.to_zarr(tmp_path / "field.zarr", zarr_format=3, consolidated=False)
    with pytest.raises(ValueError, match="not a cache store"):
        CacheView(tmp_path / "field.zarr")


# -- the arguments the dataset was constructed with ------------------------------


class Described(Windows):
    """A subclass whose arguments can all be described."""

    def __init__(self, field, days, history=(-2, -1, 0), every=1):
        view = OffsetView(
            field, ["a"], offsets={"time": np.array(history) * DAY}, unavailable="fill"
        )
        super().__init__(PointTable(time=days[::every]), {"x": view})


def test_the_arguments_are_recorded_with_the_cache(tmp_path, field):
    dataset = Described(field, TIMES, every=2)
    dataset.cache(tmp_path / "cache.zarr")
    recorded = read_arguments(tmp_path / "cache.zarr")
    described = describe(dataset.arguments)

    assert recorded.values == described.values
    assert recorded.values["every"] == 2
    assert recorded.arrays.keys() == described.arrays.keys() == {"days"}
    np.testing.assert_array_equal(recorded.arrays["days"], TIMES)
    # the field is an xarray object, which is left out
    assert recorded.left_out == described.left_out == ["field"]


def test_the_arguments_do_not_disturb_reading_the_cache(tmp_path, field):
    dataset = Described(field, TIMES)
    dataset.cache(tmp_path / "cache.zarr")
    assert_same_samples(dataset, Described.from_cache(tmp_path / "cache.zarr"))


def arguments_of(field, **changes):
    """The arguments of a `Described`, with some changed."""
    return {"field": field, "days": TIMES} | changes


def test_a_cache_is_read_with_the_arguments_it_was_written_with(tmp_path, field):
    Described(**arguments_of(field, every=2)).cache(tmp_path / "cache.zarr")
    with pytest.warns(UserWarning, match="field cannot be described"):
        cached = Described.from_cache(
            tmp_path / "cache.zarr", arguments=arguments_of(field, every=2)
        )
    # with the defaults of the constructor, as a dataset built from them has
    assert cached.arguments == Described(**arguments_of(field, every=2)).arguments


def test_a_default_given_explicitly_is_the_same_argument(tmp_path, field):
    Described(**arguments_of(field)).cache(tmp_path / "cache.zarr")
    with pytest.warns(UserWarning, match="field"):
        Described.from_cache(
            tmp_path / "cache.zarr",
            arguments=arguments_of(field, history=(-2, -1, 0), every=1),
        )


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"every": 2}, "every"),
        ({"days": TIMES[:-1]}, "days"),
        ({"history": (-1, 0)}, "history"),
    ],
)
def test_other_arguments_than_those_of_the_cache_raise(tmp_path, field, changes, named):
    Described(**arguments_of(field)).cache(tmp_path / "cache.zarr")
    with pytest.raises(ValueError, match=f"other arguments than those given: {named}"):
        Described.from_cache(
            tmp_path / "cache.zarr", arguments=arguments_of(field, **changes)
        )


def test_without_arguments_the_recorded_description_is_kept(tmp_path, field):
    dataset = Described(**arguments_of(field, every=2))
    dataset.cache(tmp_path / "cache.zarr")
    cached = Described.from_cache(tmp_path / "cache.zarr")
    assert cached.arguments == describe(dataset.arguments).values
    assert cached.arguments["every"] == 2
