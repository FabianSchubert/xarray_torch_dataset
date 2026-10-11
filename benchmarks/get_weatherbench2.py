"""Copy a subset of WeatherBench 2 to local disk, byte for byte.

Fetches three surface variables of the IFS HRES forecasts (2016 to 2022,
64 x 32 grid) and the 2 m temperature of ERA5 on the same grid, about
4.3 GB in 160,000 files. Chunking and compression stay as published.
Files that are already there are skipped, so an interrupted run can be
repeated.

The copies hold only some of the variables, so open them with
``consolidated=False``; see ``weatherbench2.py``.

usage: uv run python benchmarks/get_weatherbench2.py DEST_DIR
"""

import sys
import time
from pathlib import Path

import gcsfs

BUCKET = "weatherbench2/datasets"
STORES = {
    "hres.zarr": (
        f"{BUCKET}/hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
        ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
    ),
    "era5.zarr": (
        f"{BUCKET}/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        ["2m_temperature"],
    ),
}
COORDS = ["time", "prediction_timedelta", "latitude", "longitude", "level"]
BATCH = 2000


def copy(fs, remote, local):
    """Copy the files under `remote` that `local` does not have yet."""
    files = [f for f in fs.find(remote) if not (local / f[len(remote) + 1 :]).exists()]
    for start in range(0, len(files), BATCH):
        batch = files[start : start + BATCH]
        targets = [str(local / f[len(remote) + 1 :]) for f in batch]
        for target in targets:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        fs.get(batch, targets)
        print(f"  {local.name}: {start + len(batch)} of {len(files)}", flush=True)


if __name__ == "__main__":
    dest = Path(sys.argv[1]).expanduser()
    fs = gcsfs.GCSFileSystem(token="anon")
    for name, (remote, variables) in STORES.items():
        start = time.perf_counter()
        store = dest / name
        store.mkdir(parents=True, exist_ok=True)
        # not .zmetadata: it lists the variables that are not copied
        for meta in (".zgroup", ".zattrs"):
            fs.get(f"{remote}/{meta}", str(store / meta))
        for array in COORDS + variables:
            if fs.exists(f"{remote}/{array}"):
                copy(fs, f"{remote}/{array}", store / array)
        print(f"{name}: done in {time.perf_counter() - start:.0f} s", flush=True)
