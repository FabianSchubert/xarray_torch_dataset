"""Time single samples of an XrDataset on zarr stores, in one process.

Covers construction, correctness against in-memory data, the time per
sample for each chunking and backend, where that time goes, and the cost
of ``missing="fill"``.

usage: uv run python benchmarks/per_sample.py STORES_DIR
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import zarr

from xarray_torch_dataset import OffsetView, PointGrid, ToTensors, XrDataset

stores = Path(sys.argv[1])
HISTORY = np.arange(-3, 1)
rng = np.random.default_rng(0)


def open_store(path, backend):
    if backend == "lazy":  # xarray's own lazy indexing, no dask
        return xr.open_dataset(path, engine="zarr", chunks=None, consolidated=False)
    if backend == "dask":
        return xr.open_zarr(path, consolidated=False)
    if backend == "memory":
        return xr.open_dataset(
            path, engine="zarr", chunks=None, consolidated=False
        ).load()
    raise ValueError(backend)


def build(ds, missing="drop"):
    views = {
        "x": OffsetView(ds, ["t2m", "u10"], offsets={"time": HISTORY}, missing=missing),
        "y": OffsetView(ds, ["t2m"], missing=missing),
    }
    return XrDataset(PointGrid(time=ds.time.values), views, transform=ToTensors())


def timeit(fn, n):
    start = time.perf_counter()
    for i in range(n):
        fn(i)
    return (time.perf_counter() - start) / n * 1e3  # ms per call


def section(title):
    print(f"\n== {title}")


# ---------------------------------------------------------------------------
section("construction (validation of 8760 points, 2 views)")
for backend in ("memory", "lazy", "dask"):
    ds = open_store(stores / "chunk24.zarr", backend)
    start = time.perf_counter()
    dataset = build(ds)
    print(f"{backend:7s} {time.perf_counter() - start:6.2f} s   len={len(dataset)}")

# ---------------------------------------------------------------------------
section("correctness: zarr-backed samples equal in-memory samples")
reference = build(open_store(stores / "chunk24.zarr", "memory"))
for backend in ("lazy", "dask"):
    dataset = build(open_store(stores / "chunk24.zarr", backend))
    for i in (0, 17, 5000, len(dataset) - 1):
        a, b = dataset[i], reference[i]
        assert all(torch.equal(a[k], b[k]) for k in a)
    print(f"{backend}: ok")

# ---------------------------------------------------------------------------
section("per-sample time, random access (ms per dataset[i])")
picks = rng.integers(0, 8700, size=200)
print(f"{'':10s}" + "".join(f"{b:>10s}" for b in ("memory", "lazy", "dask")))
for chunk in (1, 24, 168):
    row = f"chunk{chunk:<5d}"
    for backend in ("memory", "lazy", "dask"):
        if backend == "memory" and chunk != 24:
            row += f"{'':>10s}"
            continue
        dataset = build(open_store(stores / f"chunk{chunk}.zarr", backend))
        dataset[0]
        row += f"{timeit(lambda i, d=dataset: d[picks[i]], len(picks)):10.2f}"
    print(row)

# ---------------------------------------------------------------------------
section("where the time goes for one sample (chunk24, lazy), ms")
ds = open_store(stores / "chunk24.zarr", "lazy")
dataset = build(ds)
view = dataset.views["x"]
point = lambda i: dataset.points[dataset.indices[picks[i]]]  # noqa: E731
n = len(picks)
print(f"select                      {timeit(lambda i: view.select(point(i)), n):7.3f}")
print(f"fetch (lazy, no read)       {timeit(lambda i: view.fetch(point(i)), n):7.3f}")
fetch_load = timeit(lambda i: view.fetch(point(i)).to_array().values, n)
print(f"fetch + load (2 vars, 4 t)  {fetch_load:7.3f}")
group = zarr.open_group(stores / "chunk24.zarr", mode="r")
arrays = [group["t2m"], group["u10"]]
raw = timeit(lambda i: [a[picks[i] : picks[i] + 4] for a in arrays], n)
print(f"raw zarr, same data, slice  {raw:7.3f}")
raw = timeit(lambda i: [a.oindex[np.arange(picks[i], picks[i] + 4)] for a in arrays], n)
print(f"raw zarr, same data, oindex {raw:7.3f}")

# ---------------------------------------------------------------------------
section('missing="fill" (chunk24, lazy), ms per fetch + load of view x')
filling = build(ds, missing="fill").views["x"]
complete = {"time": ds.time.values[5000]}
partial = {"time": ds.time.values[1]}  # two of four steps missing
absent = {"time": np.datetime64("2030-01-01")}  # everything missing
for name, p in (
    ("complete", complete),
    ("partly missing", partial),
    ("absent", absent),
):
    ms = timeit(lambda i, p=p: filling.fetch(p).to_array().values, 100)
    print(f"{name:15s} {ms:7.3f}")
