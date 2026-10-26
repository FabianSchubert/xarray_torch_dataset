"""The layout of zarr stores of samples.

An array of samples has one chunk per sample and keeps several samples in one
shard, which is one file. A write that covers a part of a shard rewrites the
shard, so samples are written a whole shard at a time.
"""

import math
import weakref

import numpy as np
import pytest
import xarray as xr
import zarr

from xarray_torch_dataset import stores
from xarray_torch_dataset.stores import (
    ZARR_FORMAT,
    SampleWriter,
    create_sample_array,
    default_samples_per_shard,
    new_group,
    new_sample_array,
)

SAMPLE = (2, 3)


def metadata_files(store) -> list[str]:
    return sorted(p.name for p in store.iterdir() if p.is_file())


def data_files(array_dir) -> list[str]:
    """The files that hold the values of an array, without its description."""
    return sorted(
        str(p.relative_to(array_dir))
        for p in array_dir.rglob("*")
        if p.is_file() and p.name != "zarr.json"
    )


def samples(n: int) -> np.ndarray:
    return np.arange(n * 6, dtype="float32").reshape(n, *SAMPLE) + 1.0


class CountingArray:
    """Passes writes on to an array and records which samples each one covered."""

    def __init__(self, array) -> None:
        self._array = array
        self.shape, self.shards = array.shape, array.shards
        self.dtype = array.dtype
        self.writes = []

    def __setitem__(self, key, value) -> None:
        self.writes.append((key.start, key.stop))
        self._array[key] = value


# ----------------------------------------------------------------------------------- #
# Groups
# ----------------------------------------------------------------------------------- #


def test_new_stores_are_in_the_format_with_shards():
    """Sharding, the reason for the move to zarr 3, needs the zarr 3 format."""
    assert ZARR_FORMAT == 3


def test_new_group_is_empty_and_in_the_format_with_shards(tmp_path):
    group = new_group(tmp_path / "store.zarr")
    assert group.metadata.zarr_format == ZARR_FORMAT
    assert list(group.keys()) == []
    assert metadata_files(tmp_path / "store.zarr") == ["zarr.json"]


def test_new_group_replaces_what_was_there(tmp_path):
    path = tmp_path / "store.zarr"
    old = new_group(path)
    old.create_array("a", shape=(3,), dtype="int64")[:] = np.arange(3)
    old.attrs["note"] = "old"

    group = new_group(path)
    assert list(group.keys()) == []
    assert dict(group.attrs) == {}


def test_arrays_of_a_group_take_its_format_and_hold_the_times(tmp_path):
    group = new_group(tmp_path / "store.zarr")
    times = np.array(["2024-01-01T00:00", "NaT"], dtype="datetime64[m]")
    group.create_array("init_time", shape=times.shape, dtype=times.dtype)[:] = times

    again = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    assert again["init_time"].metadata.zarr_format == ZARR_FORMAT
    assert again["init_time"][:].dtype == times.dtype
    assert np.isnat(again["init_time"][:][1])


# ----------------------------------------------------------------------------------- #
# Arrays of samples
# ----------------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n_samples, sample_shape, dtype, expected",
    [
        (1000, (2**20,), "float32", 256),  # 4 MB per sample
        (1000, (2**20,), "float64", 128),
        (100, (2**20,), "float32", 100),  # not more than the array holds
        (1000, (2**29,), "float32", 1),  # a sample larger than the target
        (1000, (3, 9, 390, 343), "float32", 74),  # 3 models, 9 features, a grid
        (0, (4,), "float32", 1),
    ],
)
def test_default_shard_is_about_a_gigabyte(n_samples, sample_shape, dtype, expected):
    assert stores.SHARD_BYTES == 2**30
    assert default_samples_per_shard(n_samples, sample_shape, dtype) == expected


@pytest.mark.parametrize("per_shard", [1, 4, 10, 25])
def test_sample_array_has_one_chunk_per_sample_and_one_file_per_shard(
    tmp_path, per_shard
):
    n = 10
    group = new_group(tmp_path / "store.zarr")
    array = create_sample_array(group, "fc", n, SAMPLE, "float32", np.nan, per_shard)
    array[:] = samples(n)

    again = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")["fc"]
    assert again.shape == (n, *SAMPLE)
    assert again.chunks == (1, *SAMPLE)
    assert again.shards == (per_shard, *SAMPLE)
    assert again.dtype == np.dtype("float32")
    assert again.metadata.zarr_format == ZARR_FORMAT
    np.testing.assert_array_equal(again[:], samples(n))
    assert len(data_files(tmp_path / "store.zarr" / "fc")) == math.ceil(n / per_shard)


def test_sample_array_uses_the_default_shard_size(tmp_path, monkeypatch):
    monkeypatch.setattr(stores, "SHARD_BYTES", 3 * 6 * 4)  # three samples of float32
    group = new_group(tmp_path / "store.zarr")
    assert create_sample_array(group, "a", 10, SAMPLE, "float32").shards[0] == 3
    assert new_sample_array(tmp_path / "b.zarr", 10, SAMPLE, "float32").shards[0] == 3
    # and holds the whole array where it is small enough
    assert create_sample_array(group, "c", 2, SAMPLE, "float32").shards[0] == 2


@pytest.mark.parametrize("fill_value", [None, np.nan, -1.0])
def test_samples_that_were_not_written_read_as_the_fill_value(tmp_path, fill_value):
    array = new_sample_array(tmp_path / "a.zarr", 4, SAMPLE, "float32", fill_value, 2)
    array[0:2] = samples(2)
    expected = 0.0 if fill_value is None else fill_value
    np.testing.assert_array_equal(array[2:], np.full((2, *SAMPLE), expected, "f4"))


def test_new_sample_array_is_a_store_of_its_own_and_replaces_an_older_one(tmp_path):
    path = tmp_path / "array.zarr"
    new_sample_array(path, 5, SAMPLE, "float32", samples_per_shard=2)[:] = 7.0
    array = new_sample_array(path, 3, SAMPLE, "float32", samples_per_shard=2)

    again = zarr.open_array(str(path), mode="r")
    assert again.metadata.zarr_format == ZARR_FORMAT
    assert again.shape == (3, *SAMPLE)
    assert again.shards == (2, *SAMPLE)
    assert array[:].sum() == 0.0


@pytest.mark.parametrize("per_shard", [0, -3])
def test_shard_needs_at_least_one_sample(tmp_path, per_shard):
    with pytest.raises(ValueError, match="at least one sample"):
        new_sample_array(tmp_path / "a.zarr", 4, SAMPLE, "float32", None, per_shard)
    with pytest.raises(ValueError, match="at least one sample"):
        create_sample_array(
            new_group(tmp_path / "g.zarr"), "a", 4, SAMPLE, "float32", None, per_shard
        )


def test_dimension_names_let_xarray_open_the_store(tmp_path):
    group = new_group(tmp_path / "store.zarr")
    dims = ["sample", "y", "x"]
    create_sample_array(group, "fc", 4, SAMPLE, "float32", np.nan, 2, dims)[:] = (
        samples(4)
    )
    new_sample_array(tmp_path / "a.zarr", 4, SAMPLE, "float32", dimension_names=dims)

    ds = xr.open_dataset(
        tmp_path / "store.zarr", engine="zarr", chunks=None, consolidated=False
    )
    assert ds.fc.dims == tuple(dims)
    np.testing.assert_array_equal(ds.fc.values, samples(4))
    assert zarr.open_array(str(tmp_path / "a.zarr")).metadata.dimension_names == tuple(
        dims
    )


def test_dimension_names_are_one_per_dimension(tmp_path):
    with pytest.raises(ValueError, match="has 3 dimensions, not 2"):
        new_sample_array(
            tmp_path / "a.zarr", 4, SAMPLE, "float32", dimension_names=["sample", "y"]
        )


# ----------------------------------------------------------------------------------- #
# Writing whole shards
# ----------------------------------------------------------------------------------- #


@pytest.mark.parametrize("per_shard", [1, 4, 25])
@pytest.mark.parametrize("batch", [1, 4, 7, 11])
def test_writer_stores_every_sample_whatever_the_batch_size(tmp_path, per_shard, batch):
    """Batches smaller than a shard, as large, larger and larger than the array.

    The shards hold one sample, leave a remainder, or are larger than the array.
    """
    n = 10
    array = new_sample_array(
        tmp_path / "a.zarr", n, SAMPLE, "float32", np.nan, per_shard
    )
    counting = CountingArray(array)
    data = samples(n)

    with SampleWriter(counting) as writer:
        for start in range(0, n, batch):
            writer.append(data[start : start + batch])
        assert writer.n_written == n

    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "a.zarr"))[:], data)
    # every write starts at a shard boundary and covers whole shards, except that
    # the last one ends with the array; no sample is written twice
    assert [start for start, _ in counting.writes] == [0] + [
        stop for _, stop in counting.writes[:-1]
    ]
    assert counting.writes[-1][1] == n
    assert all(start % per_shard == 0 for start, _ in counting.writes)
    assert all(stop % per_shard == 0 for _, stop in counting.writes[:-1])
    assert len(counting.writes) <= math.ceil(n / per_shard)


def test_writer_keeps_an_incomplete_shard_until_it_is_closed(tmp_path):
    array = new_sample_array(tmp_path / "a.zarr", 10, SAMPLE, "float32", np.nan, 4)
    data = samples(10)
    writer = SampleWriter(array)

    writer.append(data[:3])
    assert np.isnan(array[:]).all()
    writer.append(data[3:9])  # completes two shards, one sample is left over
    np.testing.assert_array_equal(array[:8], data[:8])
    assert np.isnan(array[8:]).all()
    writer.append(data[9:])
    assert np.isnan(array[8:]).all()

    writer.close()
    np.testing.assert_array_equal(array[:], data)
    writer.close()  # nothing is left to write
    np.testing.assert_array_equal(array[:], data)


def test_writer_stores_a_shard_as_soon_as_it_is_complete(tmp_path):
    """So that no more than one shard of samples waits in memory."""
    array = new_sample_array(tmp_path / "a.zarr", 10, SAMPLE, "float32", np.nan, 4)
    data = samples(10)
    writer = SampleWriter(array)

    writer.append(data[:4])
    np.testing.assert_array_equal(array[:4], data[:4])
    writer.append(data[4:6])
    writer.append(data[6:8])
    np.testing.assert_array_equal(array[:8], data[:8])


def test_writer_keeps_no_reference_to_the_samples_it_is_given(tmp_path):
    """A batch from a DataLoader worker keeps a file open for as long as it exists."""
    array = new_sample_array(tmp_path / "a.zarr", 10, SAMPLE, "float32", np.nan, 4)
    data = samples(10)
    writer = SampleWriter(array)
    batches = []
    for start in range(0, 10, 3):
        batch = data[start : start + 3].copy()
        batches.append(weakref.ref(batch))
        writer.append(batch)
        batch[:] = -1.0  # what waits for its shard is the writer's own
        del batch
    assert all(batch() is None for batch in batches)

    writer.close()
    np.testing.assert_array_equal(array[:], data)


def test_writer_does_not_store_a_part_of_a_shard_after_an_error(tmp_path):
    """An interrupted run leaves whole shards and nothing of the one in progress."""
    array = new_sample_array(tmp_path / "a.zarr", 10, SAMPLE, "float32", np.nan, 4)
    data = samples(10)
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        SampleWriter(array) as writer,
    ):
        writer.append(data[:6])
        raise RuntimeError("interrupted")

    np.testing.assert_array_equal(array[:4], data[:4])
    assert np.isnan(array[4:]).all()


def test_writer_of_an_array_without_shards_writes_every_batch(tmp_path):
    group = new_group(tmp_path / "g.zarr")
    array = group.create_array(
        "a", shape=(6, *SAMPLE), chunks=(1, *SAMPLE), dtype="float32", fill_value=np.nan
    )
    assert array.shards is None
    data = samples(6)
    writer = SampleWriter(array)
    writer.append(data[:2])
    np.testing.assert_array_equal(array[:2], data[:2])
    writer.append(data[2:])
    np.testing.assert_array_equal(array[:], data)


def test_writer_rejects_samples_that_do_not_fit(tmp_path):
    array = new_sample_array(tmp_path / "a.zarr", 4, SAMPLE, "float32", np.nan, 2)
    writer = SampleWriter(array)
    with pytest.raises(ValueError, match=r"shape \(3, 3\) do not fit"):
        writer.append(np.zeros((1, 3, 3), dtype="float32"))
    writer.append(samples(3))
    with pytest.raises(ValueError, match="holds 4 samples; 5 were given"):
        writer.append(samples(2))
    assert writer.n_written == 3
