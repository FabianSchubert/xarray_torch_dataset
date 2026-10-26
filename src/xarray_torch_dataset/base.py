"""
Base contracts for an xarray-backed PyTorch dataset.

    index --points--> point (a dict of coordinate labels)
    point --View----> data from one xarray source   (one View per source)
    {name: data} --transform--> returned sample

A point is a mapping from dimension names to coordinate labels, e.g.
``{"time": np.datetime64("2020-01-05")}``. Views translate a point into a
positional selection on their source; the dataset keeps only the points
that every view can serve.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import xarray as xr
from torch.utils.data import Dataset

Point = Mapping[str, Any]
Indexers = Mapping[Hashable, Any]


class Points(Protocol):
    """Any finite, indexable collection of points (a list of dicts will do)."""

    def __len__(self) -> int: ...

    def __getitem__(self, i: int, /) -> Point: ...


class View(ABC):
    """Says what to select from one xarray source for a given point.

    Subclasses implement `select`, which may contain arbitrary logic but does
    no I/O. Keeping selection separate from reading is what lets the dataset
    check up front which points a view can serve.

    A position that is ``None`` or outside the source means that data is
    missing. With ``unavailable="reject"`` the view rejects such a point.
    With ``unavailable="fill"`` it accepts every point and returns whatever
    is missing as NaN. Integers cannot hold NaN, so a filling view returns
    integer and boolean values as floats, for every point.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        variables: Sequence[str] | None = None,
        unavailable: Literal["reject", "fill"] = "reject",
    ):
        self.ds = ds if variables is None else ds[list(variables)]
        self.unavailable = unavailable

    @abstractmethod
    def select(self, point: Point) -> Indexers:
        """Return positions per dimension, as accepted by ``xr.Dataset.isel``.

        Positions are integers or integer arrays, not slices. Use `locate`
        to turn coordinate labels into positions.

        Missing data is marked by position: ``None`` for a single position,
        and a position outside the source (by convention ``-1``) for an
        entry of an array. Negative positions do not count from the end.
        Mark missing data this way rather than by raising, so that the
        shape of the selection stays known.
        """

    def locate(self, dim: Hashable, labels: Any) -> Any:
        """Positions of coordinate labels along `dim`.

        A scalar label gives a scalar position, which drops the dimension
        when selected, or ``None`` if the source has no such label, so that
        arithmetic on a failed lookup raises instead of giving a position.

        Several labels give an array of positions, with ``-1`` for labels
        the source lacks. Check for those before doing arithmetic on it.

        The labels along `dim` must be unique, since a label that occurs
        twice has no single position.
        """
        index = self.ds.indexes[dim]
        if not index.is_unique:
            repeated = index[index.duplicated()][0]
            raise ValueError(
                f"Source has repeated labels along {dim!r} (e.g. {repeated}), "
                "so a label has no single position."
            )
        if np.ndim(labels) == 0:
            try:
                return index.get_loc(labels)
            except KeyError:
                return None
        return index.get_indexer(labels)

    def labels(self, point: Point) -> dict[Hashable, Any]:
        """The coordinate labels of what the view selects for the point.

        One entry per dimension that `select(point)` selects along and that
        the source has an index for: a scalar label for a scalar position,
        an array of labels for an array of positions. Nothing is read.

        By default these are the source's labels at the selected positions,
        with a missing label (NaT, or NaN) where a position is missing. The
        labels of an integer or boolean index are floats where one is
        missing, and always with ``unavailable="fill"``, as in `fetch`.

        Override this in a view that knows what it asks for although the
        source lacks it, such as a label plus an offset that lies outside
        the source. Filling then puts these labels on the coordinates of
        the missing entries. An override returns the dimensions and shapes
        of the selection, and where a position exists, the source's label
        there; `fetch` checks this. So the coordinates of what `fetch`
        returns are always these labels, in the dtype of the coordinate.
        """
        indexers = self.select(point)
        return self._labels_of(indexers, self._exists(indexers))

    def available(self, point: Point) -> bool:
        """Whether everything that `select(point)` asks for exists.

        Unlike `accepts`, this also tells for a filling view whether
        something would be filled.
        """
        exists = self._exists(self.select(point))
        return all(e.all() for e in exists.values())

    def accepts(self, point: Point) -> bool:
        """Whether the view can serve the point, as opposed to rejecting it.

        Always with ``unavailable="fill"``; otherwise only if everything that
        `select(point)` asks for exists.
        """
        return self.unavailable == "fill" or self.available(point)

    def fetch(self, point: Point) -> xr.Dataset:
        """Read the selection into memory.

        Reading is finished when this returns, so a transform works on
        data that is loaded. Override this for views that are not a
        selection; an override has to return loaded data as well.

        Raises a ``ValueError`` if the view overrides `labels` and they do
        not agree with `select`, see `labels`.
        """
        indexers = self.select(point)
        exists = self._exists(indexers)
        labels = None  # by default taken from the positions, when needed
        if type(self).labels is not View.labels:
            labels = self.labels(point)
            self._check_labels(point, labels, indexers, exists)
        if self.unavailable == "fill":
            out = self._fill(indexers, exists, labels)
        elif all(e.all() for e in exists.values()):
            out = self.ds.isel(indexers)
        else:
            raise IndexError(f"Selection for {point!r} leaves the source.")
        return out.load()

    def _exists(self, indexers: Indexers) -> dict[Hashable, np.ndarray]:
        """Per dimension, which of the positions lie inside the source.

        ``None`` does not, and neither do negative positions, although
        ``isel`` would accept those and count from the end. A scalar
        position gives a 0-d bool array.
        """
        exists = {}
        for dim, positions in indexers.items():
            if positions is None:
                exists[dim] = np.array(False)
                continue
            if isinstance(positions, slice):
                raise TypeError(
                    f"{type(self).__name__}.select returned a slice for {dim!r}. "
                    "Return integer positions instead, e.g. np.arange(start, stop)."
                )
            positions = np.asarray(positions)
            exists[dim] = (positions >= 0) & (positions < self.ds.sizes[dim])
        return exists

    def _labels_of(
        self, indexers: Indexers, exists: Mapping[Hashable, np.ndarray]
    ) -> dict[Hashable, Any]:
        """The default `labels`, for the positions `indexers`."""
        return {
            dim: self._labels_at(dim, positions, exists[dim])
            for dim, positions in indexers.items()
            if dim in self.ds.indexes
        }

    def _check_labels(
        self,
        point: Point,
        labels: Mapping[Hashable, Any],
        indexers: Indexers,
        exists: Mapping[Hashable, np.ndarray],
    ) -> None:
        """Raise if overridden `labels` do not agree with `select`."""
        name = type(self).__name__
        dims = [dim for dim in indexers if dim in self.ds.indexes]
        if set(labels) != set(dims):
            raise ValueError(
                f"{name}.labels gives the dimensions {sorted(map(str, labels))} "
                f"for {point!r}, but select selects along {sorted(map(str, dims))}."
            )
        for dim in dims:
            wanted, present = np.asarray(labels[dim]), exists[dim]
            if wanted.shape != present.shape:
                raise ValueError(
                    f"{name}.labels gives the shape {wanted.shape} along {dim!r} "
                    f"for {point!r}, but select gives {present.shape}."
                )
            if not present.any():
                continue
            found = self.ds[dim].values[np.asarray(indexers[dim])[present]]
            if not np.array_equal(wanted[present], found):
                raise ValueError(
                    f"{name}.labels and select disagree along {dim!r} for "
                    f"{point!r}: labels {wanted[present]} where the source has "
                    f"{found}."
                )

    def _labels_at(self, dim: Hashable, positions: Any, present: np.ndarray) -> Any:
        """The labels of the source at `positions` along `dim`, see `labels`."""
        values = self.ds[dim].values
        if self.unavailable == "fill" and values.dtype.kind in "iub":
            values = values.astype(float)
        safe = np.where(present, 0 if positions is None else positions, 0)
        if present.all():
            return values[safe.astype(np.intp)]
        kind = values.dtype.kind
        dtype = float if kind in "iub" else values.dtype if kind in "fmM" else object
        blank = np.array("NaT" if kind in "mM" else np.nan, dtype=dtype)
        if not present.any():
            # also for an empty source, which has no label to stand in
            return np.full(present.shape, blank)[()]
        return np.where(present, values[safe.astype(np.intp)].astype(dtype), blank)[()]

    def _fill(
        self,
        indexers: Indexers,
        exists: Mapping[Hashable, np.ndarray],
        labels: Mapping[Hashable, Any] | None,
    ) -> xr.Dataset:
        """Read the selection, with NaN wherever a position is missing.

        The result has the shape a complete selection would have. Along a
        dimension with missing positions, the coordinate of the dimension
        gets `labels` for those entries, or the default labels if `labels`
        is ``None``: missing labels (NaN or NaT). Other coordinates along
        it are masked. A variable is only missing along dimensions it has:
        one without ``time`` is returned as it is, whatever is missing
        along ``time``.

        Integer and boolean values cannot hold NaN, so they are returned as
        floats. That holds for a complete selection too, so that the dtype
        does not depend on whether something happens to be missing.

        Works in two steps: read a selection of the right shape in which
        every missing position is replaced by a stand-in, then blank out
        the stand-ins. Example on a 31-step ``time`` axis::

            indexers  {"time": [29, 30, 31]}
            exists    {"time": [True, True, False]}
            readable  {"time": [29, 30, 29]}      # position 29 stands in
            result    time steps [step 29, step 30, NaN]
        """
        # Step 1: read. The shape is already final, but the entries at
        # missing positions hold the data of the stand-in. The stand-in is
        # a position that is read anyway, so it costs no extra I/O.
        readable = {}
        empty = set()  # dimensions along which nothing exists
        for dim, pos in indexers.items():
            present = exists[dim]
            if present.all():
                readable[dim] = pos
            elif present.any():
                stand_in = np.asarray(pos)[present].flat[0]
                readable[dim] = np.where(present, pos, stand_in)
            else:
                # Nothing exists along `dim`. Position 0 only gives the
                # selection its shape.
                scalar = pos is None or np.ndim(pos) == 0
                readable[dim] = 0 if scalar else np.zeros(np.shape(pos), dtype=int)
                empty.add(dim)
        out = self.ds.isel(readable)
        # Variables along such a dimension are missing entirely, so they
        # are not read at all.
        out = _blank(out, [n for n in out.data_vars if empty & set(self.ds[n].dims)])
        out = _as_float(out, [n for n, c in self.ds.coords.items() if c.dims])

        # Step 2: blank out the stand-ins, one dimension at a time.
        for dim, present in exists.items():
            if present.all():
                continue
            # A mask along `dim` only, which `where` broadcasts over a
            # variable's other dimensions. For a scalar position `present`
            # is 0-d, so the mask has no dimension and blanks all of it.
            mask = xr.DataArray(present, dims=[dim] * present.ndim)
            # What lies along `dim` is taken from the source, since a
            # scalar position removes `dim` from `out`. Everything else
            # is left alone: `where` on the whole dataset would give those
            # variables the dimension `dim` and blank them as well.
            along_dim = [n for n, v in self.ds.variables.items() if dim in v.dims]
            # `where` keeps values where the mask is True and puts NaN
            # elsewhere. The coordinates are masked too; otherwise a
            # missing entry would carry the label of the stand-in. The
            # index of `dim` gets the labels that were asked for instead.
            labelled = dim in self.ds.indexes
            out = out.assign(
                {n: out[n].where(mask) for n in along_dim if n in out.data_vars}
            )
            out = out.assign_coords(
                {
                    n: out[n].where(mask).variable
                    for n in along_dim
                    if n in out.coords and not (labelled and n == dim)
                }
            )
            if labelled:
                # It keeps its dtype, which must not depend on the sample.
                if labels is None:
                    labels = self._labels_of(indexers, exists)
                wanted = np.asarray(labels[dim])
                if out[dim].dtype.kind in "fmM":
                    wanted = wanted.astype(out[dim].dtype)
                out = out.assign_coords({dim: (out[dim].dims, wanted)})
        return out


def _blank(ds: xr.Dataset, names: Sequence[Hashable]) -> xr.Dataset:
    """`ds` with the data variables `names` replaced by NaN (or NaT), unread."""
    blank = {}
    for name in names:
        var = ds[name]
        if var.dtype.kind in "iubf":
            values = np.full(var.shape, np.nan)
            if var.dtype.kind == "f":
                values = values.astype(var.dtype)
        elif var.dtype.kind in "mM":
            values = np.full(var.shape, "NaT", dtype=var.dtype)
        else:
            continue  # left to be read and masked
        blank[name] = (var.dims, values, var.attrs)
    return ds.assign(blank) if blank else ds


def _as_float(ds: xr.Dataset, coords: Sequence[Hashable]) -> xr.Dataset:
    """`ds` with integer and boolean data variables and `coords` as floats."""
    variables = {
        n: v.astype(float) for n, v in ds.data_vars.items() if v.dtype.kind in "iub"
    }
    if variables:
        ds = ds.assign(variables)
    as_float = {n: ds[n].astype(float) for n in coords if ds[n].dtype.kind in "iub"}
    return ds.assign_coords(as_float) if as_float else ds


class XrDataset(Dataset):
    """Serves, for each accepted point, ``{name: view.fetch(point)}``.

    The data of each view is an ``xr.Dataset`` that is loaded into memory.
    `transform` receives that dict and returns the sample. A DataLoader
    needs a transform that returns tensors or arrays, such as `ToTensors`,
    because torch cannot batch xarray objects.

    The transform can come in two stages: `prepare` does the expensive part
    and returns one ``xr.Dataset`` per sample, the prepared sample, and
    `transform` turns that into the sample. `cache` stores the prepared
    samples, and `from_cache` reads them back, so that the same `transform`
    serves both.

    Points that some view rejects are dropped at construction, or
    reported if ``rejected="raise"``. `n_rejected` says how many points
    each view rejected; a point counts for the first view that rejected it.

    ``validate=False`` skips that check, which takes time on many points.
    Every point is then kept, `n_rejected` is ``None``, and a point that a
    view would have rejected raises an ``IndexError`` when it is fetched.
    """

    def __init__(
        self,
        points: Points,
        views: Mapping[str, View],
        transform: Callable | None = None,
        prepare: Callable[[dict[str, xr.Dataset]], xr.Dataset] | None = None,
        rejected: Literal["drop", "raise"] = "drop",
        validate: bool = True,
    ):
        self.points = points
        self.views = views
        self.transform = transform
        self.prepare = prepare
        if not len(points):
            raise ValueError("There are no points.")
        # positions in `points` of the points the dataset serves
        self.indices: np.ndarray
        self.n_rejected: dict[str, int] | None
        if not validate:
            self.indices = np.arange(len(points))
            self.n_rejected = None
            return

        accepted = []
        self.n_rejected = dict.fromkeys(views, 0)
        for i in range(len(points)):
            point = points[i]
            for name, view in views.items():
                if not view.accepts(point):
                    self.n_rejected[name] += 1
                    break
            else:
                accepted.append(i)
        # an array, not a list: it is compact, and forked DataLoader
        # workers share it instead of copying it
        self.indices = np.array(accepted, dtype=np.int64)

        n_rejected = len(points) - len(accepted)
        if not accepted or (rejected == "raise" and n_rejected):
            how_many = f"{n_rejected} of" if accepted else "All"
            by_view = ", ".join(
                f"{n} by view {name!r}" for name, n in self.n_rejected.items() if n
            )
            raise ValueError(
                f"{how_many} {len(points)} points were rejected: {by_view}. "
                "(Each point is counted for the first view that rejected it.)"
            )

    def labels(self, name: str) -> dict[Hashable, np.ndarray]:
        """The labels that view `name` selects for every point, without reading.

        Per dimension an array with one row per sample, in the order of the
        samples, followed by the shape of the labels of one point: shape
        (n_samples,) for a scalar label, (n_samples, window) for a window.
        See `View.labels`.
        """
        view = self.views[name]
        per_point = [view.labels(self.points[int(i)]) for i in self.indices]
        return {dim: np.stack([p[dim] for p in per_point]) for dim in per_point[0]}

    def available(self, name: str) -> np.ndarray:
        """Whether view `name` has all data for every sample, without reading.

        A bool array of shape (n_samples,). See `View.available`.
        """
        view = self.views[name]
        return np.array(
            [view.available(self.points[int(i)]) for i in self.indices], dtype=bool
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        point = self.points[int(self.indices[idx])]
        data = {name: view.fetch(point) for name, view in self.views.items()}
        sample = self.prepare(data) if self.prepare else data
        return self.transform(sample) if self.transform else sample

    def cache(self, path: str | Path, **options: Any) -> None:
        """Write the prepared sample of every point to a zarr store at `path`.

        See `xarray_torch_dataset.cache.write_cache` for the layout of the
        store and the `options`. Needs `prepare`.
        """
        from xarray_torch_dataset.cache import write_cache

        write_cache(self, path, **options)

    @classmethod
    def from_cache(
        cls,
        path: str | Path,
        transform: Callable | None = None,
        points: Points | None = None,
        rejected: Literal["drop", "raise"] = "drop",
    ) -> "XrDataset":
        """A dataset of the prepared samples that `cache` wrote to `path`.

        Its samples are ``transform(prepared sample)``, so the `transform` of
        the dataset that was cached gives the same samples. `points` reads
        part of the cache, by default all of it; a point the cache lacks is
        rejected as by any view. The result is an `XrDataset`, also when
        called on a subclass, whose constructor needs sources that a cached
        dataset does not have.
        """
        from xarray_torch_dataset.cache import CachedRow, CacheView

        view = CacheView(path)
        return XrDataset(
            view.points if points is None else points,
            {"cache": view},
            transform=transform,
            prepare=CachedRow(view.point_dims),
            rejected=rejected,
            # the points of the cache are all in it
            validate=points is not None,
        )
