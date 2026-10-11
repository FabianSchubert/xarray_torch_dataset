"""Time an XrDataset on real forecast data from WeatherBench 2.

The dataset pairs IFS HRES forecasts with ERA5 as the target. Points are
on a 6-hourly grid of init times, finer than the 12-hourly forecasts, so
`FindLatestForecastView` has to pick an earlier forecast with a longer
lead time for every other point. The target is ERA5 at the valid time.

usage: uv run python benchmarks/weatherbench2.py SOURCE [cold]
  SOURCE  ``remote`` reads from the public bucket; a directory reads the
          copy made by ``get_weatherbench2.py``.
  cold    for a local copy: drop it from the page cache before each
          measurement, so that the reads come from disk.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset import (
    FindLatestForecastView,
    PointGrid,
    ToTensors,
    View,
    XrDataset,
)

BUCKET = "gs://weatherbench2/datasets"
REMOTE = {
    "hres.zarr": f"{BUCKET}/hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
    "era5.zarr": f"{BUCKET}/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
}
FORECAST = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
BATCH = 32
HOUR = np.timedelta64(1, "h")


class ValidTimeView(View):
    """An analysis at the point's valid time (init time plus lead time)."""

    def select(self, point):
        return {"time": self.locate("time", point["init_time"] + point["lead_time"])}


def open_store(source, name):
    if source == "remote":
        kwargs = {"storage_options": {"token": "anon"}}
        path = REMOTE[name]
    else:
        # the copy holds only some variables, which .zmetadata would not match
        kwargs = {"consolidated": False}
        path = str(Path(source).expanduser() / name)
    return xr.open_dataset(
        path, engine="zarr", chunks=None, decode_timedelta=True, **kwargs
    )


def build(source, validate=True):
    hres = open_store(source, "hres.zarr").rename(
        time="init_time", prediction_timedelta="lead_time"
    )
    era5 = open_store(source, "era5.zarr")
    # two years of 6-hourly init times; ERA5 ends with 2021, so the last
    # points have no target
    init_times = np.arange("2020-01-01", "2022-01-01", 6 * HOUR, dtype="datetime64[h]")
    lead_times = np.arange(6, 78, 6) * HOUR
    views = {
        "forecast": FindLatestForecastView(
            hres, FORECAST, init_time_offset_limits=(-6 * HOUR, 0 * HOUR)
        ),
        "target": ValidTimeView(era5, ["2m_temperature"]),
    }
    return XrDataset(
        PointGrid(init_time=init_times.astype("datetime64[ns]"), lead_time=lead_times),
        views,
        transform=ToTensors(),
        validate=validate,
    )


def evict(source):
    """Drop the files of a local copy from the page cache."""
    os.sync()
    for path in Path(source).expanduser().rglob("*"):
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


def throughput(dataset, workers, seconds):
    """Samples per second of a shuffled DataLoader, over about `seconds`."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=workers,
        multiprocessing_context="forkserver" if workers else None,
        generator=torch.Generator().manual_seed(0),
    )
    batches = iter(loader)
    next(batches)  # worker start-up is not counted
    start, n = time.perf_counter(), 0
    while time.perf_counter() - start < seconds:
        next(batches)
        n += 1
    return n * BATCH / (time.perf_counter() - start)


if __name__ == "__main__":
    source = sys.argv[1]
    cold = len(sys.argv) > 2 and sys.argv[2] == "cold"
    if cold and source == "remote":
        sys.exit("cold only applies to a local copy")
    prepare = (lambda: evict(source)) if cold else (lambda: None)
    print(f"source: {source}{' (cold)' if cold else ''}")

    start = time.perf_counter()
    dataset = build(source)
    seconds = time.perf_counter() - start
    n_points = len(dataset.points)
    print(f"\n== construction: {seconds:.1f} s for {n_points} points and 2 views")
    print(f"kept {len(dataset)}, rejected {dataset.n_rejected}")
    indices = dataset.indices

    print("\n== one sample (ms), random access, no workers")
    picks = np.random.default_rng(0).integers(0, len(dataset), size=30)
    prepare()
    start = time.perf_counter()
    for i in picks:
        dataset[i]
    print(f"{(time.perf_counter() - start) / len(picks) * 1e3:.1f}")

    print(f"\n== DataLoader, shuffled, batch_size={BATCH} (samples per second)")
    print("".join(f"{w:>4d} workers" for w in (0, 4, 8)))
    row = ""
    for workers in (0, 4, 8):
        # reuse the validation from above to save start-up time
        dataset = build(source, validate=False)
        dataset.indices = indices
        prepare()
        row += f"{throughput(dataset, workers, seconds=20):12.0f}"
    print(row, flush=True)
