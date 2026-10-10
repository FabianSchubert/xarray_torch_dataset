"""DataLoader throughput, and one read per batch against one per sample.

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

HISTORY = np.arange(-3, 1)
BATCH = 32


def open_store(path, backend):
    if backend == "lazy":
        return xr.open_dataset(path, engine="zarr", chunks=None, consolidated=False)
    return xr.open_zarr(path, consolidated=False)


class PerSample(XrDataset):
    """An XrDataset that a DataLoader reads sample by sample."""

    __getitems__ = None  # type: ignore[assignment]


def build(ds, batched=True):
    views = {
        "x": OffsetView(ds, ["t2m", "u10"], offsets={"time": HISTORY}),
        "y": OffsetView(ds, ["t2m"]),
    }
    cls = XrDataset if batched else PerSample
    return cls(PointGrid(time=ds.time.values), views, transform=ToTensors())


def timeit(fn, n):
    start = time.perf_counter()
    for i in range(n):
        fn(i)
    return (time.perf_counter() - start) / n * 1e3


def loaders(stores, backends):
    print(f"\n== DataLoader, shuffled, batch_size={BATCH} (samples per second)")
    print(f"{'':27s}" + "".join(f"{w:>4d} workers" for w in (0, 4, 8)))
    configs = [(c, b, r) for c in (1, 24, 168) for b in backends for r in (False, True)]
    for chunk, backend, batched in configs:
        read = "one read" if batched else "per sample"
        row = f"chunk{chunk:<4d} {backend:5s} {read:10s}"
        for workers in (0, 4, 8):
            ds = open_store(stores / f"chunk{chunk}.zarr", backend)
            dataset = build(ds, batched)
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


def batched_row(ds, rng):
    """Times of four ways to read the history of BATCH random points, in ms."""
    view = OffsetView(ds, ["t2m", "u10"], offsets={"time": HISTORY})
    batches = [
        [{"time": t} for t in ds.time.values[rng.integers(3, 8760, size=BATCH)]]
        for _ in range(5)
    ]

    def per_sample(i):
        return np.stack([view.fetch(p).to_array().values for p in batches[i]])

    def positions(i):
        return np.stack([view.select(p)["time"] for p in batches[i]])

    def vectorized(i):
        # one pointwise selection with a (batch, time) indexer
        indexer = xr.DataArray(positions(i), dims=("batch", "time"))
        out = view.ds.isel(time=indexer).to_array().values
        return np.moveaxis(out, 1, 0)  # batch first

    def unique_take(i):
        # read every needed step once, sorted, then gather in memory
        pos = positions(i)
        steps, inverse = np.unique(pos, return_inverse=True)
        block = view.ds.isel(time=steps).to_array().values  # var, step, ...
        return np.moveaxis(block[:, inverse.reshape(pos.shape)], 1, 0)

    def fetch_many(i):
        return np.stack([d.to_array().values for d in view.fetch_many(batches[i])])

    assert np.array_equal(per_sample(0), fetch_many(0))
    assert np.array_equal(per_sample(0), vectorized(0))
    assert np.array_equal(per_sample(0), unique_take(0))
    fns = (per_sample, vectorized, unique_take, fetch_many)
    return [timeit(fn, len(batches)) for fn in fns]


def batched(stores, backends):
    rng = np.random.default_rng(0)
    print(f"\n== one read per batch instead of per sample ({BATCH} random points), ms")
    names = ("per sample", "vectorized", "unique+take", "fetch_many")
    print(f"{'':20s}" + "".join(f"{name:>12s}" for name in names))
    for chunk in (1, 24, 168):
        for backend in backends:
            ds = open_store(stores / f"chunk{chunk}.zarr", backend)
            times = "".join(f"{ms:12.1f}" for ms in batched_row(ds, rng))
            print(f"chunk{chunk:<4d} {backend:9s}{times}", flush=True)


if __name__ == "__main__":
    stores, backends = Path(sys.argv[1]), sys.argv[2].split(",")
    loaders(stores, backends)
    batched(stores, backends)
