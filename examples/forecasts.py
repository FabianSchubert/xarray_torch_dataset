import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset import (
    FindLatestForecastView,
    PointGrid,
    ToTensors,
    XrDataset,
)


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
                unavailable="fill",
            ),
        },
        transform=ToTensors(),
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))

    print(f"{len(dataset)} of {len(points)} points are kept")
    print({k: v.shape for k, v in batch.items()})
