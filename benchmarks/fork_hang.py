"""Does a zarr-backed XrDataset survive DataLoader workers?

With the dask backend and forked workers it does not, if the parent has
read from the store before the workers start: the first batch never
arrives. DASK_SCHEDULER=synchronous avoids that, as do spawn and forkserver.

usage: uv run python benchmarks/fork_hang.py STORE BACKEND START_METHOD TOUCH
  STORE         one store, e.g. STORES_DIR/chunk24.zarr
  BACKEND       lazy (xr.open_dataset without dask) or dask (xr.open_zarr)
  START_METHOD  fork, spawn or forkserver
  TOUCH         1 reads one sample in the parent first, 0 does not
"""

import faulthandler
import sys
import time

import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset import OffsetView, PointGrid, ToTensors, XrDataset

path, backend, method, touch = sys.argv[1:5]
faulthandler.enable()


def main():
    if backend == "lazy":
        ds = xr.open_dataset(path, engine="zarr", chunks=None, consolidated=False)
    else:
        ds = xr.open_zarr(path, consolidated=False)
    views = {"x": OffsetView(ds, ["t2m", "u10"], offsets={"time": np.arange(-3, 1)})}
    dataset = XrDataset(PointGrid(time=ds.time.values), views, transform=ToTensors())
    if touch == "1":
        dataset[0]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        multiprocessing_context=method,
        timeout=20,
    )
    start = time.perf_counter()
    it = iter(loader)
    next(it)
    first = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(20):
        next(it)
    rate = 20 * 32 / (time.perf_counter() - start)
    print(f"ok    first batch {first:5.2f} s, then {rate:6.0f} samples/s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAIL  {type(e).__name__}: {str(e)[:90]}")
