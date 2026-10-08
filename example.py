"""
Minimal mockup: building a dataset from points, views and a transform,
including one custom view with non-contiguous selection logic.
"""

import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset import OffsetView, PointGrid, View, XrDataset

# --------------------------------------------------------------------------
# Example user implementations
# --------------------------------------------------------------------------


class SameWeekdayView(View):
    """The same weekday in each of the `weeks` weeks before the point."""

    def __init__(self, ds, variables, weeks):
        super().__init__(ds, variables)
        self.weeks = weeks

    def select(self, point):
        back = np.arange(self.weeks, 0, -1) * np.timedelta64(7, "D")
        return {"time": point["time"] - back}


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

    # Every second day. Points too close to the start to have a full history
    # are dropped by the dataset; no offset needs to be matched by hand.
    points = PointGrid(time=inputs.time.values[::2])

    dataset = XrDataset(
        points=points,
        views={
            "x": OffsetView(inputs, ["t2m", "u10"], offsets={"time": range(-3, 1)}),
            "x_weekly": SameWeekdayView(inputs, ["t2m"], weeks=2),
            "y": OffsetView(targets, ["precip"]),
        },
        transform=ToTensors(),
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    print(f"{len(dataset)} of {len(points)} points are valid")
    print({k: v.shape for k, v in batch.items()})
