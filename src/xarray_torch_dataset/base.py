"""
Base classes (the contracts) for an xarray-backed PyTorch dataset.

    index --Sampler--> point (in the Anchor's sampling space)
    point --View-----> data from one xarray source   (one View per source)
    {name: data} --Transform--> returned sample

A "point" is whatever the user's Anchor/Sampler/Views agree on
(e.g. a dict of coordinates). The framework never looks inside it.
"""

from abc import ABC, abstractmethod

import xarray as xr
from torch.utils.data import Dataset


class Anchor(ABC):
    """The sampling space: a finite, indexable set of points."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, i: int): ...


class Sampler(ABC):
    """Maps a dataset index to a point in the anchor's space."""

    def __init__(self, anchor: Anchor):
        self.anchor = anchor

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int): ...


class View(ABC):
    """Knows what to pull from one xarray source for a given point."""

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    @abstractmethod
    def __call__(self, point): ...


class XrDataset(Dataset):
    def __init__(self, sampler: Sampler, views: dict, transform=None):
        self.sampler = sampler
        self.views = views
        self.transform = transform

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        point = self.sampler[idx]
        sample = {name: view(point) for name, view in self.views.items()}
        return self.transform(sample) if self.transform else sample
