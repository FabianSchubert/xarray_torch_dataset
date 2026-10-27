"""Caching prepared samples in a zarr store, and reading them back as a view.

An `XrDataset` serves ``transform(prepare(data))``. `XrDataset.cache`
writes ``prepare(data)`` of every point with `write_cache`, and
`XrDataset.from_cache` reads them back through a `CacheView`, so that a
subclass and its ``from_cache`` give the same samples.
"""

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset.base import Indexers, Point, View, XrDataset
from xarray_torch_dataset.points import PointTable
from xarray_torch_dataset.stores import (
    SampleWriter,
    create_sample_array,
    new_group,
)

# the dimension along which a cache holds its samples
SAMPLE_DIM = "sample"

# the attribute of a cache store that says which of its arrays are what
CACHE_ATTRIBUTE = "xarray_torch_dataset"

# Variables of at most this many bytes per sample, such as times or flags,
# are held in memory while writing and stored as one chunk. Larger ones get
# one chunk per sample, in shards.
SMALL_SAMPLE_BYTES = 1024


def write_cache(
    dataset: XrDataset,
    path: str | Path,
    attrs: Mapping[str, Any] | None = None,
    samples_per_shard: int | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    multiprocessing_context: str | None = None,
    overwrite: bool = False,
) -> None:
    """Write the prepared sample of every point of `dataset` to `path`.

    `XrDataset.cache` calls this. The `prepare` of `dataset` returns an
    ``xr.Dataset`` per sample, the prepared sample, whose variables have the
    same names, dimensions, shapes and dtypes for every sample. Each data
    variable becomes an array with a leading ``sample`` dimension, in the
    order of the samples. Coordinates are written once, and must be the same
    for every sample; something that differs between samples is a data
    variable. The labels of the points are written as one array per
    dimension of the points, along ``sample``.
    `attrs` become the attributes of the store.

    Variables of at most `SMALL_SAMPLE_BYTES` per sample are stored as one
    chunk; larger ones get one chunk per sample, and `samples_per_shard`
    samples per file (see `create_sample_array`). Samples are read with a
    DataLoader of `batch_size` and `num_workers`, and written a whole shard
    at a time. `multiprocessing_context` is the DataLoader's start method of
    the workers; ``"forkserver"`` avoids forking a process in which zarr or
    dask already run threads, and needs a dataset whose class and views
    can be pickled and imported. An existing store at `path` raises a
    ``FileExistsError`` unless `overwrite` is set.
    """
    if type(dataset).prepare is XrDataset.prepare and not dataset.cached:
        raise ValueError(
            "Only a dataset that overrides `prepare` can be cached: what prepare "
            "returns is what the cache stores."
        )
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists.")
    template = _checked(dataset._prepared(0))
    point_dims = _point_dims(dataset)
    names = [*template.data_vars, *template.coords, *point_dims]
    clash = {str(n) for n in names if names.count(n) > 1}
    if clash:
        raise ValueError(
            f"The points and the prepared samples both have {sorted(clash)}."
        )
    if SAMPLE_DIM in template.dims:
        raise ValueError(f"Prepared samples must not have a {SAMPLE_DIM!r} dimension.")

    n = len(dataset)
    root = new_group(path)
    root.attrs.update(
        {
            **(attrs or {}),
            CACHE_ATTRIBUTE: {
                "points": point_dims,
                "coords": [str(c) for c in template.coords],
            },
        }
    )

    # the points and the coordinates, which are known before reading
    for dim in point_dims:
        labels = np.array([dataset.points[int(i)][dim] for i in dataset.indices])
        _write_whole(root, dim, labels, [SAMPLE_DIM], {})
    for name, coord in template.coords.items():
        _write_whole(
            root, str(name), coord.values, list(map(str, coord.dims)), coord.attrs
        )

    # the data variables, sample by sample
    writers: dict[Hashable, SampleWriter] = {}
    held: dict[Hashable, np.ndarray] = {}
    for name, var in template.data_vars.items():
        if var.dtype.itemsize * var.size <= SMALL_SAMPLE_BYTES:
            held[name] = np.empty((n, *var.shape), dtype=var.dtype)
            continue
        array = create_sample_array(
            root,
            str(name),
            n,
            var.shape,
            var.dtype,
            _missing(var.dtype),
            samples_per_shard,
            [SAMPLE_DIM, *map(str, var.dims)],
        )
        array.attrs.update(var.attrs)
        writers[name] = SampleWriter(array)

    loader = torch.utils.data.DataLoader(
        _PreparedSamples(dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_Stack(template),
        multiprocessing_context=multiprocessing_context if num_workers else None,
    )
    start = 0
    for batch in loader:
        stop = start + len(next(iter(batch.values())))
        for name, values in batch.items():
            if name in writers:
                writers[name].append(values)
            else:
                held[name][start:stop] = values
        start = stop
    for writer in writers.values():
        writer.close()
    for name, values in held.items():
        var = template[name]
        _write_whole(
            root, str(name), values, [SAMPLE_DIM, *map(str, var.dims)], var.attrs
        )


class CacheView(View):
    """The prepared samples of a cache store that `write_cache` wrote.

    `select` finds the row of a point by its labels; a point the cache does
    not hold is missing, and rejected unless ``unavailable="fill"``. The
    fetched row is the prepared sample with the point's labels as scalar
    coordinates. `points` are the points of the cache, in their order, and
    `attrs` the attributes that were given to `write_cache`.
    """

    def __init__(
        self,
        path: str | Path,
        variables: Sequence[str] | None = None,
        unavailable: Literal["reject", "fill"] = "reject",
    ):
        ds = xr.open_dataset(path, engine="zarr", chunks=None, consolidated=False)
        layout = ds.attrs.pop(CACHE_ATTRIBUTE, None)
        if layout is None:
            raise ValueError(f"{path} is not a cache store written by write_cache.")
        self.point_dims: list[str] = list(layout["points"])
        ds = ds.set_coords([*self.point_dims, *layout["coords"]])
        super().__init__(ds, variables, unavailable)
        self.attrs = dict(ds.attrs)
        columns = {dim: ds[dim].values for dim in self.point_dims}
        self.points = PointTable(**columns)
        # the row of every point, by its labels in the dtypes of the store
        self._dtypes = {dim: values.dtype for dim, values in columns.items()}
        self._rows = {
            key: row
            for row, key in enumerate(
                zip(*(values.tolist() for values in columns.values()), strict=True)
            )
        }
        if len(self._rows) < len(self.points):
            raise ValueError(f"{path} holds some points more than once.")

    def prepared_sample(self, row: xr.Dataset) -> xr.Dataset:
        """The prepared sample in the fetched row of a point.

        That is the row without the labels of the point and the attributes of
        the store, which belong to the store, see `attrs`.
        """
        sample = row.drop_vars(self.point_dims)
        sample.attrs = {}
        return sample

    def select(self, point: Point) -> Indexers:
        key = tuple(
            np.asarray(point[dim]).astype(self._dtypes[dim]).item()
            for dim in self.point_dims
        )
        return {SAMPLE_DIM: self._rows.get(key)}


class _PreparedSamples(torch.utils.data.Dataset):
    """The prepared samples of a dataset, for a DataLoader to read."""

    def __init__(self, dataset: XrDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        return self.dataset._prepared(idx)


def _checked(sample: Any) -> xr.Dataset:
    """The prepared sample, checked to be an ``xr.Dataset``."""
    if not isinstance(sample, xr.Dataset):
        raise TypeError(
            "A cache needs a dataset whose prepare returns an xr.Dataset per "
            f"sample, not a {type(sample).__name__}."
        )
    return sample


def _point_dims(dataset: XrDataset) -> list[str]:
    """The dimensions of the points of `dataset`, from its first point."""
    return list(dataset.points[int(dataset.indices[0])])


def _missing(dtype: np.dtype) -> Any:
    """The fill value that reads as missing, or ``None`` for the default."""
    if dtype.kind == "f":
        return np.nan
    if dtype.kind in "mM":
        return np.array("NaT", dtype=dtype)[()]
    return None


def _write_whole(
    group: Any, name: str, values: np.ndarray, dims: list[str], attrs: Mapping
) -> None:
    """Write `values` as one array of a single chunk."""
    values = np.asarray(values)
    array = group.create_array(
        name,
        shape=values.shape,
        chunks=values.shape if values.size else None,
        dtype=values.dtype,
        fill_value=_missing(values.dtype),
        dimension_names=dims,
    )
    array.attrs.update(attrs)
    array[...] = values


class _Stack:
    """Collates prepared samples into a batch: per data variable, an array.

    It checks every sample against the first one, so that the store holds
    what every sample has. A class rather than a function, so that the
    workers of a DataLoader can receive it.
    """

    def __init__(self, template: xr.Dataset):
        self.template = template

    def __call__(self, samples: list[Any]) -> dict[Hashable, np.ndarray]:
        for sample in samples:
            self._check(_checked(sample))
        return {
            name: np.stack([sample[name].values for sample in samples])
            for name in self.template.data_vars
        }

    def _check(self, sample: xr.Dataset) -> None:
        template = self.template
        if set(sample.data_vars) != set(template.data_vars):
            names = sorted(map(str, sample.data_vars))
            first = sorted(map(str, template.data_vars))
            raise ValueError(
                f"A prepared sample has the data variables {names}, "
                f"the first one {first}."
            )
        for name, var in template.data_vars.items():
            other = sample[name]
            if (other.dims, other.shape, other.dtype) != (
                var.dims,
                var.shape,
                var.dtype,
            ):
                raise ValueError(
                    f"{name!r} of a prepared sample has the dimensions {other.dims}, "
                    f"shape {other.shape} and dtype {other.dtype}; in the first one "
                    f"{var.dims}, {var.shape} and {var.dtype}."
                )
        if set(sample.coords) != set(template.coords) or not all(
            sample[c].variable.equals(template[c].variable) for c in template.coords
        ):
            raise ValueError(
                "The coordinates of a prepared sample differ from those of the "
                "first one. Something that differs between samples has to be a "
                "data variable."
            )
