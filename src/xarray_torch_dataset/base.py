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

    def accepts(self, point: Point) -> bool:
        """Whether the view can serve the point, as opposed to rejecting it.

        Always with ``unavailable="fill"``; otherwise only if everything that
        `select(point)` asks for exists.
        """
        if self.unavailable == "fill":
            return True
        exists = self._exists(self.select(point))
        return all(e.all() for e in exists.values())

    def fetch(self, point: Point) -> xr.Dataset:
        """Read the selection. Override for views that are not a selection."""
        indexers = self.select(point)
        return self._take(self.ds, indexers, self._exists(indexers), point)

    def fetch_many(self, points: Sequence[Point]) -> list[xr.Dataset]:
        """What `fetch` gives for each of the points, from a single read.

        Reads every position that some point needs once, into memory, and
        takes each point's selection from that block. The results are the
        same as from `fetch`, except that they are already loaded.

        Along several dimensions the block is the outer product of the
        positions needed along each. If that is more data than the points
        ask for (as for points that each need one scattered element), the
        points are fetched one by one instead. The same holds for views
        that override `fetch`.
        """
        if type(self).fetch is not View.fetch:
            return [self.fetch(point) for point in points]
        if not points:
            return []
        selections = [self.select(point) for point in points]
        exists = [self._exists(indexers) for indexers in selections]

        # Per dimension that every point selects along, the sorted positions
        # that exist and are needed by some point. Other dimensions are read
        # in full, as they would be for a single point.
        shared = set(selections[0]).intersection(*selections[1:])
        steps = {}
        for dim in shared:
            needed = [
                np.asarray(indexers[dim])[e[dim]].ravel()
                for indexers, e in zip(selections, exists, strict=True)
                if indexers[dim] is not None
            ]
            steps[dim] = np.unique(np.concatenate(needed)) if needed else np.array([])
            if not steps[dim].size:
                # nothing exists along `dim`; `_fill` needs a stand-in to read
                steps[dim] = np.array([0])

        block_size = np.prod([len(s) for s in steps.values()])
        requested = sum(np.prod([e[dim].sum() for dim in shared]) for e in exists)
        if block_size > requested:
            return [self.fetch(point) for point in points]
        block = self.ds.isel(steps).load()

        out = []
        for point, indexers, e in zip(points, selections, exists, strict=True):
            # the same selection, as positions in the block
            in_block = {
                dim: pos
                if dim not in steps or pos is None
                else np.searchsorted(steps[dim], np.where(e[dim], pos, steps[dim][0]))
                for dim, pos in indexers.items()
            }
            out.append(self._take(block, in_block, e, point))
        return out

    def _take(
        self,
        source: xr.Dataset,
        indexers: Indexers,
        exists: Mapping[Hashable, np.ndarray],
        point: Point,
    ) -> xr.Dataset:
        """Select from `source`, filling or raising if positions are missing.

        `exists` says which positions are missing; where it is False the
        entry of `indexers` is not used.
        """
        if self.unavailable == "fill":
            return self._fill(source, indexers, exists)
        if all(e.all() for e in exists.values()):
            return source.isel(indexers)
        raise IndexError(f"Selection for {point!r} leaves the source.")

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

    def _fill(
        self,
        source: xr.Dataset,
        indexers: Indexers,
        exists: Mapping[Hashable, np.ndarray],
    ) -> xr.Dataset:
        """Read the selection, with NaN wherever a position is missing.

        The result has the shape a complete selection would have. Along a
        dimension with missing positions, the coordinates of those entries
        are masked as well (NaN or NaT). A variable is only missing along
        dimensions it has: one without ``time`` is returned as it is,
        whatever is missing along ``time``.

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
        out = source.isel(readable)
        # Variables along such a dimension are missing entirely, so they
        # are not read at all.
        out = _blank(out, [n for n in out.data_vars if empty & set(source[n].dims)])
        out = _as_float(out, [n for n, c in source.coords.items() if c.dims])

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
            along_dim = [n for n, v in source.variables.items() if dim in v.dims]
            # `where` keeps values where the mask is True and puts NaN
            # elsewhere. The coordinates are masked too; otherwise a
            # missing entry would carry the label of the stand-in.
            out = out.assign(
                {n: out[n].where(mask) for n in along_dim if n in out.data_vars}
            )
            out = out.assign_coords(
                {n: out[n].where(mask).variable for n in along_dim if n in out.coords}
            )
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

    Points that some view rejects are dropped at construction, or
    reported if ``rejected="raise"``. `n_rejected` says how many points
    each view rejected; a point counts for the first view that rejected it.

    ``validate=False`` skips that check, which takes time on many points.
    Every point is then kept, `n_rejected` is ``None``, and a point that a
    view would have rejected raises an ``IndexError`` when it is fetched.

    ``batched=True`` is experimental. A DataLoader that batches then asks
    for a whole batch at once, which costs one read per view instead of one
    per sample (see `View.fetch_many`). On chunked stores that can be
    faster, mostly with few workers. One point that cannot be fetched then
    fails its whole batch.
    """

    def __init__(
        self,
        points: Points,
        views: Mapping[str, View],
        transform: Callable | None = None,
        rejected: Literal["drop", "raise"] = "drop",
        validate: bool = True,
        batched: bool = False,
    ):
        self.points = points
        self.views = views
        self.transform = transform
        self.batched = batched
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        point = self.points[int(self.indices[idx])]
        sample = {name: view.fetch(point) for name, view in self.views.items()}
        return self.transform(sample) if self.transform else sample

    @property
    def __getitems__(self):
        """What torch's DataLoader calls with all indices of a batch, if set.

        ``None`` unless ``batched=True``, so that by default a DataLoader
        asks for one sample at a time through ``__getitem__``.
        """
        return self._getitems if self.batched else None

    def _getitems(self, indices):
        """The samples at `indices`, as a list, from one read per view."""
        points = [self.points[int(self.indices[idx])] for idx in indices]
        fetched = {name: view.fetch_many(points) for name, view in self.views.items()}
        samples = [
            {name: data[k] for name, data in fetched.items()}
            for k in range(len(points))
        ]
        return [self.transform(s) for s in samples] if self.transform else samples
