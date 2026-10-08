"""
Base contracts for an xarray-backed PyTorch dataset.

    index --points--> point (a dict of coordinate labels)
    point --View----> data from one xarray source   (one View per source)
    {name: data} --transform--> returned sample

A point is a mapping from dimension names to coordinate labels, e.g.
``{"time": np.datetime64("2020-01-05")}``. Views translate a point into a
label-based selection on their source; the dataset keeps only the points
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
    """

    def __init__(self, ds: xr.Dataset, variables: Sequence[str] | None = None):
        self.ds = ds if variables is None else ds[list(variables)]

    @abstractmethod
    def select(self, point: Point) -> Indexers:
        """Return label-based indexers, as accepted by ``xr.Dataset.sel``.

        Raise a ``LookupError`` if the point cannot be served.
        """

    def is_valid(self, point: Point) -> bool:
        """Whether `select(point)` only refers to labels the source has.

        Slices are not checked: label slices never fail in xarray, they
        silently return fewer elements.
        """
        try:
            indexers = self.select(point)
        except LookupError:
            return False
        for dim, labels in indexers.items():
            if isinstance(labels, slice):
                continue
            if (self.ds.indexes[dim].get_indexer(np.atleast_1d(labels)) < 0).any():
                return False
        return True

    def fetch(self, point: Point) -> xr.Dataset:
        """Read the selection. Override for views that are not a selection."""
        return self.ds.sel(self.select(point))


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
