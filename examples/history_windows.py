"""
Windows along a time axis: a dataset built from points, views and a
transform. The inputs are a four-step history and the same weekday in the
two previous weeks; the target is the point's own time step.
"""

import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset import (
    OffsetView,
    PointGrid,
    SameWeekdayView,
    ToTensors,
    XrDataset,
)

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
    print(f"{len(dataset)} of {len(points)} points are kept")
    print({k: v.shape for k, v in batch.items()})
