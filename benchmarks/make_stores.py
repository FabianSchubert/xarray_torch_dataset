"""Write the same synthetic dataset to zarr with different time chunkings.

One year of hourly data on a 128 x 128 grid, two float32 variables (about
1 GB), as ``chunk1.zarr``, ``chunk24.zarr`` and ``chunk168.zarr``. The other
benchmarks read these stores.

usage: uv run python benchmarks/make_stores.py STORES_DIR
"""

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

times = np.arange("2023-01-01", "2024-01-01", dtype="datetime64[h]")  # 8760
shape = (len(times), 128, 128)
dims = ("time", "lat", "lon")
rng = np.random.default_rng(0)

ds = xr.Dataset(
    {
        "t2m": (dims, rng.standard_normal(shape, dtype=np.float32)),
        "u10": (dims, rng.standard_normal(shape, dtype=np.float32)),
    },
    coords={
        "time": times.astype("datetime64[ns]"),
        "lat": np.linspace(-60, 60, 128),
        "lon": np.linspace(0, 357, 128),
    },
)
print(ds)

for time_chunk in (1, 24, 168):
    path = out / f"chunk{time_chunk}.zarr"
    if path.exists():
        continue
    start = time.perf_counter()
    encoding = {v: {"chunks": (time_chunk, 128, 128)} for v in ds.data_vars}
    ds.to_zarr(path, encoding=encoding, zarr_format=3, consolidated=False)
    print(f"{path.name}: {time.perf_counter() - start:.1f}s")
