from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset.base import Indexers, Point, View, XrDataset
from xarray_torch_dataset.points import PointGrid


def fancy_noise(shape, betas, rng=None):
    rng = np.random.default_rng(rng)
    white = rng.standard_normal(shape)
    axes = list(betas)
    X = np.fft.fftn(white, axes=axes)
    for ax, beta in betas.items():
        f = np.abs(np.fft.fftfreq(shape[ax]))
        f[0] = f[1]
        filt = f ** (-beta / 2)
        bshape = [1] * len(shape)
        bshape[ax] = shape[ax]
        X *= filt.reshape(bshape)
    out = np.fft.ifftn(X, axes=axes).real
    return out / out.std()


class FindLatestForecastView(View):
    """The latest forecast for the point's valid time.

    A point has an init time and a lead time, which add up to a valid time.
    Among the source's init times within `init_time_offset_limits` of the
    point's init time (lower and upper offset, ``None`` for no limit), the
    view selects the most recent one that has a lead time reaching the same
    valid time.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        variables: Sequence[str] | None = None,
        init_time_offset_limits: tuple[np.timedelta64 | None, np.timedelta64 | None]
        | None = None,
        missing: Literal["drop", "fill"] = "drop",
    ) -> None:
        super().__init__(ds, variables, missing)
        self.init_times = self.ds.init_time.values

        # a limit of None means unbounded on that side
        lower, upper = init_time_offset_limits or (None, None)
        self.init_time_offset_limits = (
            np.timedelta64(-1_000_000, "D") if lower is None else lower,
            np.timedelta64(1_000_000, "D") if upper is None else upper,
        )

    def select(self, point: Point) -> Indexers:
        valid_time = point["init_time"] + point["lead_time"]

        # positions of the init times inside the window around the point's
        lower, upper = self.init_time_offset_limits
        offsets = self.init_times - point["init_time"]
        candidates = np.flatnonzero((offsets >= lower) & (offsets <= upper))

        # the lead time each candidate needs to reach the valid time, as a
        # position along lead_time (-1 where the source has no such lead)
        leads = self.locate("lead_time", valid_time - self.init_times[candidates])
        found = leads >= 0
        if not found.any():
            # None marks a missing position, so the point is dropped or
            # filled with NaN
            return {"init_time": None, "lead_time": None}

        candidates, leads = candidates[found], leads[found]
        latest = np.argmax(self.init_times[candidates])
        return {"init_time": candidates[latest], "lead_time": leads[latest]}


class ToTensors(torch.nn.Module):
    """Stack each source's variables into one tensor (variables first)."""

    def forward(self, sample):
        return {
            name: torch.from_numpy(ds.to_array().values).float()
            for name, ds in sample.items()
        }


if __name__ == "__main__":
    init_times_1h = np.arange(
        "2024-10-01T00:00", "2024-10-03T00:00", dtype="datetime64[h]"
    )
    init_times_3h = init_times_1h[::3].copy()

    lead_times = np.arange(1, 25).astype("timedelta64[h]")

    dims = ("init_time", "lead_time", "lat", "lon")

    shape_1h = (len(init_times_1h), len(lead_times), 256, 256)
    shape_3h = (len(init_times_3h), len(lead_times), 256, 256)

    fc_1h = xr.Dataset(
        {"t2m": (dims, fancy_noise(shape_1h, {2: 1.0, 3: 1.0}))},
        coords={"init_time": init_times_1h, "lead_time": lead_times},
    )

    fc_3h = xr.Dataset(
        {"t2m": (dims, fancy_noise(shape_3h, {2: 1.0, 3: 1.0}))},
        coords={"init_time": init_times_3h, "lead_time": lead_times},
    )

    print("Forecast 1h:")
    print(fc_1h)
    print("Forecast 3h:")
    print(fc_3h)

    points = PointGrid(init_time=init_times_1h, lead_time=lead_times)

    dataset = XrDataset(
        points=points,
        views={
            "fc_1h": FindLatestForecastView(
                fc_1h,
                ["t2m"],
                init_time_offset_limits=(
                    np.timedelta64(-2, "h"),
                    np.timedelta64(0, "h"),
                ),
            ),
            "fc_3h": FindLatestForecastView(
                fc_3h,
                ["t2m"],
                init_time_offset_limits=(
                    np.timedelta64(-2, "h"),
                    np.timedelta64(0, "h"),
                ),
                # points without a matching 3h forecast are kept, as NaN
                missing="fill",
            ),
        },
        transform=ToTensors(),
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))

    print(f"{len(dataset)} of {len(points)} points are valid")
    print({k: v.shape for k, v in batch.items()})
