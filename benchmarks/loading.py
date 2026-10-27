"""DataLoader throughput on the synthetic zarr stores.

Workers are started with forkserver: with the dask backend, forked workers
hang once the parent has read from the store (see ``fork_hang.py``).

usage: uv run python benchmarks/loading.py STORES_DIR BACKEND[,BACKEND]
  BACKEND is ``lazy`` (xr.open_dataset without dask) or ``dask`` (xr.open_zarr).
  Set DASK_SCHEDULER=synchronous to time dask without its thread pool.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset import OffsetView, PointGrid, ToTensors, XrDataset


class Tensors(XrDataset):
    """The data of the views, as tensors."""

    def transform(self, sample):
        return ToTensors()(sample)


HISTORY = np.arange(-3, 1) * np.timedelta64(1, "h")
BATCH = 32


def open_store(path, backend):
    if backend == "lazy":
        return xr.open_dataset(path, engine="zarr", chunks=None, consolidated=False)
    return xr.open_zarr(path, consolidated=False)


def build(ds):
    views = {
        "x": OffsetView(ds, ["t2m", "u10"], offsets={"time": HISTORY}),
        "y": OffsetView(ds, ["t2m"]),
    }
    points = PointGrid(time=ds.time.values)
    return Tensors(points, views)


def loaders(stores, backends):
    print(f"\n== DataLoader, shuffled, batch_size={BATCH} (samples per second)")
    print(f"{'':16s}" + "".join(f"{w:>4d} workers" for w in (0, 4, 8)))
    for chunk in (1, 24, 168):
        for backend in backends:
            row = f"chunk{chunk:<4d} {backend:5s}"
            for workers in (0, 4, 8):
                dataset = build(open_store(stores / f"chunk{chunk}.zarr", backend))
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=BATCH,
                    shuffle=True,
                    num_workers=workers,
                    multiprocessing_context="forkserver" if workers else None,
                    generator=torch.Generator().manual_seed(0),
                )
                n_batches = 8 if workers == 0 else 40
                it = iter(loader)
                next(it)  # worker start-up is not counted
                start = time.perf_counter()
                for _ in range(n_batches):
                    next(it)
                row += f"{n_batches * BATCH / (time.perf_counter() - start):12.0f}"
                del it, loader
            print(row, flush=True)


if __name__ == "__main__":
    loaders(Path(sys.argv[1]), sys.argv[2].split(","))
