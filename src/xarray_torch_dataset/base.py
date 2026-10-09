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

    With ``missing="drop"`` a point is invalid if anything it selects is
    missing. With ``missing="fill"`` every point is valid and whatever is
    missing is returned as NaN.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        variables: Sequence[str] | None = None,
        missing: Literal["drop", "fill"] = "drop",
    ):
        self.ds = ds if variables is None else ds[list(variables)]
        self.missing = missing

    @abstractmethod
    def select(self, point: Point) -> Indexers:
        """Return positional indexers, as accepted by ``xr.Dataset.isel``.

        Use `locate` to turn coordinate labels into positions. Positions
        outside the source mark missing data, by convention ``-1``: negative
        ones do not count from the end. Returning such positions, rather
        than raising, keeps the shape of the selection known, which
        ``missing="fill"`` needs.

        A ``LookupError`` also marks the point as one the view cannot
        serve, but only works with ``missing="drop"``.
        """

    def locate(self, dim: Hashable, labels: Any) -> Any:
        """Positions of coordinate labels along `dim`.

        A scalar label gives a scalar position, which drops the dimension
        when selected. A missing label raises a ``LookupError`` with
        ``missing="drop"`` and gives position ``-1`` with ``missing="fill"``.
        """
        index = self.ds.indexes[dim]
        fill = self.missing == "fill"
        if np.ndim(labels) == 0:
            try:
                return index.get_loc(labels)
            except KeyError:
                if fill:
                    return -1
                raise
        labels = np.asarray(labels)
        positions = index.get_indexer(labels)
        if not fill and (positions < 0).any():
            raise LookupError(f"Source has no {dim!r} labels {labels[positions < 0]}.")
        return positions

    def is_valid(self, point: Point) -> bool:
        """Whether the view can serve the point.

        With ``missing="drop"``: `select(point)` succeeds and stays inside
        the source. With ``missing="fill"``: always, provided `select`
        does not raise.
        """
        try:
            indexers = self.select(point)
        except LookupError as err:
            if self.missing == "fill":
                raise ValueError(
                    f"{type(self).__name__}.select raised for {point!r}. With "
                    'missing="fill" it has to return position -1 for missing '
                    "data instead, so that the shape of the fill is known."
                ) from err
            return False
        return self.missing == "fill" or self._in_bounds(indexers)

    def _in_bounds(self, indexers: Indexers) -> bool:
        """Whether all positions lie in ``[0, size)`` of their dimension.

        ``isel`` would wrap negative positions and silently shorten slices
        that leave the source, so both count as out of bounds here. Slice
        bounds may be ``None`` or equal the size.
        """
        for dim, positions in indexers.items():
            size = self.ds.sizes[dim]
            if isinstance(positions, slice):
                bounds = (positions.start, positions.stop)
                if any(b is not None and not 0 <= b <= size for b in bounds):
                    return False
                continue
            positions = np.asarray(positions)
            if positions.size and (positions.min() < 0 or positions.max() >= size):
                return False
        return True

    def fetch(self, point: Point) -> xr.Dataset:
        """Read the selection. Override for views that are not a selection."""
        indexers = self.select(point)
        if self._in_bounds(indexers):
            return self.ds.isel(indexers)
        if self.missing == "fill":
            return self._fetch_filled(indexers)
        raise IndexError(f"Selection for {point!r} leaves the source.")

    def _fetch_filled(self, indexers: Indexers) -> xr.Dataset:
        """Read the selection, with NaN wherever a position is out of bounds.

        The result has the shape a complete selection would have. Along a
        dimension with missing positions, the coordinates of those entries
        are masked as well (NaN or NaT).

        Works in two steps: read a selection of the right shape in which
        every missing position is replaced by a stand-in, then blank out
        the stand-ins. Example on a 31-step ``time`` axis::

            indexers  {"time": [-1, 0, 1, 31]}
            in_source {"time": [False, True, True, False]}
            readable  {"time": [0, 0, 1, 0]}      # position 0 stands in
            result    time steps [NaN, step 0, step 1, NaN]
        """
        # Step 1: per dimension, find which positions exist in the source
        # and build indexers that `isel` can read without wrapping.
        readable = {}  # dim -> positions, with 0 in place of missing ones
        in_source = {}  # dim -> bool per position; only dims with gaps
        for dim, positions in indexers.items():
            size = self.ds.sizes[dim]
            if isinstance(positions, slice):
                # `isel` would cut a slice off at the edge of the source.
                # Spelling it out keeps the positions beyond the edge.
                positions = self._slice_positions(positions, size)
            # A scalar position becomes a 0-d array here: `exists` is then a
            # single bool, and `isel` still drops the dimension.
            positions = np.asarray(positions)
            exists = (positions >= 0) & (positions < size)
            readable[dim] = np.where(exists, positions, 0)
            if not exists.all():
                in_source[dim] = exists

        # Step 2: read. The shape is already final, but the entries at
        # missing positions hold the data of position 0.
        out = self.ds.isel(readable)

        # Step 3: blank out those entries, one dimension at a time.
        for dim, exists in in_source.items():
            # A mask along `dim` only, which `where` broadcasts over all
            # other dimensions. For a scalar position `exists` is 0-d, so
            # the mask has no dimension and blanks the whole sample.
            mask = xr.DataArray(exists, dims=[dim] * exists.ndim)
            # `where` keeps values where the mask is True and puts NaN
            # elsewhere, but only in data variables...
            out = out.where(mask)
            # ...so the coordinates along `dim` (taken from the source,
            # since a scalar position removes `dim` from `out`) are masked
            # separately. Otherwise a missing entry would carry the label
            # of position 0.
            along_dim = [n for n, c in self.ds.coords.items() if dim in c.dims]
            out = out.assign_coords(
                {name: out[name].where(mask).variable for name in along_dim}
            )
        return out

    @staticmethod
    def _slice_positions(positions: slice, size: int) -> np.ndarray:
        """The positions a slice stands for, including those out of bounds.

        ``slice(28, 33)`` gives ``[28, 29, 30, 31, 32]`` whatever the size.
        Open bounds run to the end of the source in the direction of the
        step. Negative bounds are positions before the start, not counted
        from the end.
        """
        step = positions.step or 1
        if step > 0:
            default_start, default_stop = 0, size
        else:
            default_start, default_stop = size - 1, -1
        start = default_start if positions.start is None else positions.start
        stop = default_stop if positions.stop is None else positions.stop
        return np.arange(start, stop, step)


class XrDataset(Dataset):
    """Serves, for each valid point, ``{name: view.fetch(point)}``.

    Points that some view cannot serve are dropped at construction, or
    reported if ``invalid="raise"``.
    """

    def __init__(
        self,
        points: Points,
        views: Mapping[str, View],
        transform: Callable | None = None,
        invalid: Literal["drop", "raise"] = "drop",
    ):
        self.points = points
        self.views = views
        self.transform = transform
        self.indices = [
            i
            for i in range(len(points))
            if all(view.is_valid(points[i]) for view in views.values())
        ]
        n_invalid = len(points) - len(self.indices)
        if not self.indices:
            raise ValueError("No point can be served by all views.")
        if invalid == "raise" and n_invalid:
            raise ValueError(f"{n_invalid} of {len(points)} points are invalid.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        point = self.points[self.indices[idx]]
        sample = {name: view.fetch(point) for name, view in self.views.items()}
        return self.transform(sample) if self.transform else sample
