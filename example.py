"""
Minimal mockup: xarray-backed PyTorch dataset built from small, swappable parts.

    index --Sampler--> point (in the Anchor's sampling space)
    point --View-----> data from one xarray source   (one View per source)
    {name: data} --Transform--> returned sample

A "point" is whatever the user's Anchor/Sampler/Views agree on
(e.g. a dict of coordinates). The framework never looks inside it.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
# Base classes (the contracts)
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Example user implementations
# --------------------------------------------------------------------------

class TimeAnchor(Anchor):
    """Every time step of a reference dataset is a valid point."""

    def __init__(self, times):
        self.times = times

    def __len__(self):
        return len(self.times)

    def __getitem__(self, i):
        return {"time": self.times[i]}


class StrideSampler(Sampler):
    """Every `stride`-th anchor point, skipping the first `offset`."""

    def __init__(self, anchor, stride=1, offset=0):
        super().__init__(anchor)
        self.stride, self.offset = stride, offset

    def __len__(self):
        return (len(self.anchor) - self.offset) // self.stride

    def __getitem__(self, idx):
        return self.anchor[self.offset + idx * self.stride]


class HistoryView(View):
    """The last `n` time steps of some variables, up to and including the point."""

    def __init__(self, ds, variables, n):
        super().__init__(ds)
        self.variables, self.n = variables, n

    def __call__(self, point):
        end = self.ds.indexes["time"].get_loc(point["time"])
        return self.ds[self.variables].isel(time=slice(end - self.n + 1, end + 1))


class AtTimeView(View):
    """Variables at exactly the point's time."""

    def __init__(self, ds, variables):
        super().__init__(ds)
        self.variables = variables

    def __call__(self, point):
        return self.ds[self.variables].sel(time=point["time"])


class ToTensors(torch.nn.Module):
    """Stack each source's variables into one tensor (variables first)."""

    def forward(self, sample):
        return {
            name: torch.from_numpy(ds.to_array().values).float()
            for name, ds in sample.items()
        }


# --------------------------------------------------------------------------
# Usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    times = np.arange("2020-01-01", "2020-02-01", dtype="datetime64[D]")
    shape = (len(times), 8, 8)
    dims = ("time", "lat", "lon")

    inputs = xr.Dataset(
        {"t2m": (dims, np.random.rand(*shape)), "u10": (dims, np.random.rand(*shape))},
        coords={"time": times},
    )  # e.g. xr.open_zarr("inputs.zarr")
    targets = xr.Dataset(
        {"precip": (dims, np.random.rand(*shape))},
        coords={"time": times},
    )  # e.g. xr.open_zarr("targets.zarr")

    history = 4
    anchor = TimeAnchor(inputs.time.values)
    sampler = StrideSampler(anchor, stride=2, offset=history - 1)

    dataset = XrDataset(
        sampler=sampler,
        views={
            "x": HistoryView(inputs, ["t2m", "u10"], n=history),
            "y": AtTimeView(targets, ["precip"]),
        },
        transform=ToTensors(),
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    print(len(dataset), {k: v.shape for k, v in batch.items()})
