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
    missing. With ``missing="drop"`` such a point is invalid. With
    ``missing="fill"`` every point is valid and whatever is missing is
    returned as NaN.
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
        """
        index = self.ds.indexes[dim]
        if np.ndim(labels) == 0:
            try:
                return index.get_loc(labels)
            except KeyError:
                return None
        return index.get_indexer(labels)

    def is_valid(self, point: Point) -> bool:
        """Whether the view can serve the point.

        Always with ``missing="fill"``; otherwise only if everything that
        `select(point)` asks for exists.
        """
        if self.missing == "fill":
            return True
        exists = self._exists(self.select(point))
        return all(e.all() for e in exists.values())

    def fetch(self, point: Point) -> xr.Dataset:
        """Read the selection. Override for views that are not a selection."""
        indexers = self.select(point)
        exists = self._exists(indexers)
        if all(e.all() for e in exists.values()):
            return self.ds.isel(indexers)
        if self.missing == "fill":
            return self._fill(indexers, exists)
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
        self, indexers: Indexers, exists: Mapping[Hashable, np.ndarray]
    ) -> xr.Dataset:
        """Read the selection, with NaN wherever a position is missing.

        The result has the shape a complete selection would have. Along a
        dimension with missing positions, the coordinates of those entries
        are masked as well (NaN or NaT).

        Works in two steps: read a selection of the right shape in which
        every missing position is replaced by a stand-in, then blank out
        the stand-ins. Example on a 31-step ``time`` axis::

            indexers  {"time": [-1, 0, 1, 31]}
            exists    {"time": [False, True, True, False]}
            readable  {"time": [0, 0, 1, 0]}      # position 0 stands in
            result    time steps [NaN, step 0, step 1, NaN]
        """
        # Step 1: read. The shape is already final, but the entries at
        # missing positions hold the data of position 0.
        readable = {
            dim: 0 if pos is None else np.where(exists[dim], pos, 0)
            for dim, pos in indexers.items()
        }
        out = self.ds.isel(readable)

        # Step 2: blank out those entries, one dimension at a time.
        for dim, present in exists.items():
            if present.all():
                continue
            # A mask along `dim` only, which `where` broadcasts over all
            # other dimensions. For a scalar position `present` is 0-d, so
            # the mask has no dimension and blanks the whole sample.
            mask = xr.DataArray(present, dims=[dim] * present.ndim)
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
