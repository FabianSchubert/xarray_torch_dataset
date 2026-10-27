"""
Windows along a time axis: a dataset built from points, views and a
transform, the method of a subclass. The inputs are the last four days and
the same weekday in the two previous weeks; the target is the point's own
day.
"""

import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset import (
    OffsetView,
    PointGrid,
    ToTensors,
    XrDataset,
)


class Windows(XrDataset):
    """The windows of the views, as tensors."""

    def transform(self, sample):
        return ToTensors()(sample)


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

    # Offsets are durations: they are added to the point's time, and the
    # result is looked up in the source.
    day = np.timedelta64(1, "D")
    last_four_days = np.arange(-3, 1) * day
    same_weekday = np.array([-14, -7]) * day

    dataset = Windows(
        points=points,
        views={
            "x": OffsetView(inputs, ["t2m", "u10"], offsets={"time": last_four_days}),
            "x_weekly": OffsetView(inputs, ["t2m"], offsets={"time": same_weekday}),
            "y": OffsetView(targets, ["precip"]),
        },
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    print(f"{len(dataset)} of {len(points)} points are kept")
    print({k: v.shape for k, v in batch.items()})
