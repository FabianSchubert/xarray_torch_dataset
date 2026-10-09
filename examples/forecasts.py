from collections.abc import Hashable, Sequence
from typing import Any, Literal

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


class FindForecastView(View):
    def __init__(
        self,
        ds: xr.Dataset,
        variables: Sequence[str] | None = None,
        init_time_offset_limits: tuple[np.timedelta64 | None, np.timedelta64 | None]
        | None = None,
        missing: Literal["drop", "fill"] = "drop",
    ) -> None:
        super().__init__(ds, variables, missing)

        self.init_times = ds.init_time.values
        self.lead_times = ds.lead_time.values
        self.valid_times = (
            self.init_times[:, np.newaxis] + self.lead_times[np.newaxis, :]
        )

        # a limit of None means unbounded on that side
        lower, upper = init_time_offset_limits or (None, None)
        self.init_time_offset_limits = (
            np.timedelta64(-1_000_000, "D") if lower is None else lower,
            np.timedelta64(1_000_000, "D") if upper is None else upper,
        )

    def select(self, point: Point) -> Indexers:
        indexers: dict[Hashable, Any] = {}

        ref_init_time = point["init_time"]
        ref_lead_time = point["lead_time"]
        ref_valid_time = ref_init_time + ref_lead_time

        it_offset = self.init_times[:, np.newaxis] - ref_init_time
        init_idx, lead_idx = np.where(
            (ref_valid_time == self.valid_times)
            & (it_offset >= self.init_time_offset_limits[0])
            & (it_offset <= self.init_time_offset_limits[1])
        )

        if len(init_idx) == 0:
            # no forecast in the init time window: None marks a missing
            # position, so the point is dropped or filled with NaN
            return {"init_time": None, "lead_time": None}

        # get the matching candidate with the most recent init time
        # (init_idx and lead_idx are parallel arrays of candidate positions)
        most_recent_candidate = np.argmax(self.init_times[init_idx])
        # note that we are not checking for whether there are duplicate
        # hits
        indexers["init_time"] = init_idx[most_recent_candidate]
        indexers["lead_time"] = lead_idx[most_recent_candidate]

        return indexers


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
            "fc_1h": FindForecastView(
                fc_1h,
                ["t2m"],
                init_time_offset_limits=(
                    np.timedelta64(-2, "h"),
                    np.timedelta64(0, "h"),
                ),
            ),
            "fc_3h": FindForecastView(
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
