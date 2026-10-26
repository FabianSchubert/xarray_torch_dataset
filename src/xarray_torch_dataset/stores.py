"""The layout of zarr stores of samples.

An array of samples has one chunk per sample, which is what is read at a
time, and keeps several chunks in one shard, which is what is stored as a
file. A store of one chunk per sample would otherwise consist of one file per
sample. A write that covers a part of a shard reads the shard and writes it
again, so samples are written a whole shard at a time, with a `SampleWriter`.

Stores are written in the zarr 3 format, which sharding needs, and are read
with plain zarr or xarray calls.
"""

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import numpy as np
import zarr
from numpy.typing import DTypeLike

# the version of the zarr format in which new stores are written
ZARR_FORMAT: Literal[3] = 3

# The size of a shard before compression that the default number of samples
# per shard aims at. Large shards mean few files, which is what sharding is
# used for here.
SHARD_BYTES = 2**30


def new_group(path: str | Path) -> zarr.Group:
    """An empty group as the root of a store, replacing what is at `path`.

    Arrays of samples are added with `create_sample_array`, other arrays with
    the group's own ``create_array``.
    """
    return zarr.open_group(str(path), mode="w", zarr_format=ZARR_FORMAT)


def default_samples_per_shard(
    n_samples: int, sample_shape: tuple[int, ...], dtype: DTypeLike
) -> int:
    """As many samples as make up `SHARD_BYTES` before compression.

    At least one, and not more than the array holds.
    """
    sample_bytes = int(np.prod(sample_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    return max(1, min(n_samples, SHARD_BYTES // max(sample_bytes, 1)))


def _sample_layout(
    n_samples: int,
    sample_shape: tuple[int, ...],
    dtype: DTypeLike,
    samples_per_shard: int | None,
    dimension_names: Sequence[str] | None,
) -> dict[str, Any]:
    """Shape, chunks, shards, dtype and dimension names for ``create_array``.

    `samples_per_shard` of ``None`` chooses it from the size of a sample.
    """
    sample_shape = tuple(int(n) for n in sample_shape)
    if samples_per_shard is None:
        samples_per_shard = default_samples_per_shard(n_samples, sample_shape, dtype)
    if samples_per_shard < 1:
        raise ValueError(f"A shard holds at least one sample, not {samples_per_shard}.")
    if dimension_names is not None and len(dimension_names) != 1 + len(sample_shape):
        raise ValueError(
            f"An array of samples of shape {sample_shape} has "
            f"{1 + len(sample_shape)} dimensions, not {len(dimension_names)}: "
            f"{list(dimension_names)}."
        )
    return {
        "shape": (n_samples, *sample_shape),
        "chunks": (1, *sample_shape),
        "shards": (int(samples_per_shard), *sample_shape),
        "dtype": dtype,
        "dimension_names": None if dimension_names is None else list(dimension_names),
    }


def create_sample_array(
    group: zarr.Group,
    name: str,
    n_samples: int,
    sample_shape: tuple[int, ...],
    dtype: DTypeLike,
    fill_value: Any = None,
    samples_per_shard: int | None = None,
    dimension_names: Sequence[str] | None = None,
) -> zarr.Array:
    """Add an array of samples to a group: one chunk per sample, in shards.

    The array has the shape ``(n_samples, *sample_shape)``. Samples that have
    not been written read as `fill_value`, by default the zero of the dtype.
    `samples_per_shard` samples are stored in one file, by default as many as
    make up about 1 GB before compression. `dimension_names`, one per
    dimension, the samples' first, let xarray open the array. Fill the array
    with a `SampleWriter`.
    """
    return group.create_array(
        name,
        fill_value=fill_value,
        **_sample_layout(
            n_samples, sample_shape, dtype, samples_per_shard, dimension_names
        ),
    )


def new_sample_array(
    path: str | Path,
    n_samples: int,
    sample_shape: tuple[int, ...],
    dtype: DTypeLike,
    fill_value: Any = None,
    samples_per_shard: int | None = None,
    dimension_names: Sequence[str] | None = None,
) -> zarr.Array:
    """A store that is a single array of samples, replacing what is at `path`.

    The arguments are those of `create_sample_array`.
    """
    return zarr.create_array(
        store=str(path),
        fill_value=fill_value,
        zarr_format=ZARR_FORMAT,
        overwrite=True,
        **_sample_layout(
            n_samples, sample_shape, dtype, samples_per_shard, dimension_names
        ),
    )


class SampleWriter:
    """Writes samples to an array in their order, a whole shard at a time.

    Samples are collected until a shard is complete and then written in one
    piece, whatever the size of the batches they arrive in. The last shard,
    which may not be full, is written by `close`, or on leaving the ``with``
    block::

        with SampleWriter(array) as writer:
            for batch in batches:
                writer.append(batch)

    The array is filled from its first sample on. An array without shards is
    written batch by batch.

    Up to one shard of samples is held in memory, as a copy: the writer keeps
    no reference to the arrays it is given, so that they are released as soon
    as the caller drops them. This matters for batches from the workers of a
    DataLoader, each of which keeps a file open for as long as it exists. If
    writing is interrupted, the samples of the shard in progress are not in
    the store. Only one writer may write to an array.

    `n_written` is the number of samples appended so far.
    """

    def __init__(self, array: zarr.Array) -> None:
        self._array = array
        # without shards, every sample is a file of its own and is written as
        # it comes
        self._per_shard = 1 if array.shards is None else array.shards[0]
        # the samples of the shard in progress, of which the first `_n_held`
        # are set; allocated when the first sample has to wait
        self._held: np.ndarray | None = None
        self._n_held = 0
        self._n_stored = 0
        self.n_written = 0

    def append(self, samples: np.ndarray) -> None:
        """Add the next samples, of shape ``(n, *sample shape)``.

        Raises a ``ValueError`` if they do not have the shape of the array's
        samples, or the array has no room for them.
        """
        samples = np.asarray(samples)
        if samples.shape[1:] != self._array.shape[1:]:
            raise ValueError(
                f"Samples of shape {samples.shape[1:]} do not fit an array of "
                f"samples of shape {self._array.shape[1:]}."
            )
        if self.n_written + len(samples) > self._array.shape[0]:
            raise ValueError(
                f"The array holds {self._array.shape[0]} samples; "
                f"{self.n_written + len(samples)} were given."
            )
        self.n_written += len(samples)
        while len(samples) > 0:
            if self._n_held == 0 and len(samples) >= self._per_shard:
                # whole shards that start at a shard boundary need not wait
                n = len(samples) - len(samples) % self._per_shard
                self._store(samples[:n])
            else:
                n = min(len(samples), self._per_shard - self._n_held)
                self._hold(samples[:n])
                if self._n_held == self._per_shard:
                    self.close()
            samples = samples[n:]

    def _hold(self, samples: np.ndarray) -> None:
        """Copy samples to the shard in progress, at most as many as it lacks."""
        if self._held is None:
            self._held = np.empty(
                (min(self._per_shard, self._array.shape[0]), *self._array.shape[1:]),
                dtype=self._array.dtype,
            )
        self._held[self._n_held : self._n_held + len(samples)] = samples
        self._n_held += len(samples)

    def _store(self, samples: np.ndarray) -> None:
        """Write samples after those that are stored.

        They are a whole number of shards, except at the end of the array.
        """
        self._array[self._n_stored : self._n_stored + len(samples)] = samples
        self._n_stored += len(samples)

    def close(self) -> None:
        """Write the samples of the last shard, which may not be full."""
        if self._held is not None and self._n_held > 0:
            self._store(self._held[: self._n_held])
            self._n_held = 0

    def __enter__(self) -> "SampleWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Write what is pending, unless the block was left by an error."""
        if exc_type is None:
            self.close()
