# Project status

Snapshot as of 2026-10-26. This file records where the project stands and which design questions are open, so work can continue on another machine.

## What this project is

An experiment in building PyTorch datasets on top of xarray from small, swappable parts:

```
index --points--> point (a dict of coordinate labels)
point --View----> data from one xarray source   (one View per source)
{name: data} --transform--> returned sample
```

A point is a mapping from dimension names to coordinate labels, for example `{"time": np.datetime64("2020-01-05")}`. Views translate a point into a positional selection on their source, and the dataset keeps only the points that every view can serve.

## Current state

The package lives in `src/xarray_torch_dataset/`; everything below is re-exported from the package root.

- **`Point`** (`base.py`): type alias for a dict of coordinate labels.
- **`Points`** (`base.py`): a protocol, not a base class. Anything with `__len__` and `__getitem__(i) -> Point` qualifies, so a plain list of dicts works.
- **`View`** (`base.py`): abstract. Subclasses implement `select(point)`, which returns positional indexers as accepted by `xr.Dataset.isel` and does no I/O; it may contain arbitrary logic. Positions are integers or integer arrays; slices are not supported. Missing data is marked by position, and that is the only way to mark it: `None` for a single position, and a position outside the source (by convention `-1`) for an entry of an array. Negative positions never count from the end, and any exception from `select` is treated as a bug. What a view does with a point whose data the source cannot fully supply is set per view by `unavailable`:
  - `unavailable="reject"` (the default): the view rejects the point. What happens to rejected points is then up to the dataset (`rejected`, below).
  - `unavailable="fill"`: the view accepts every point, and `fetch` returns data of the full expected shape with NaN at the missing positions (so a window can be partly filled). The coordinate of a dimension with missing entries shows the labels of `labels(point)` there, missing labels (NaN or NaT) unless the view overrides `labels`; other coordinates along it are masked. Integer and boolean variables and coordinates come back as floats, for complete samples as well, so the dtype does not depend on the sample. Only positions that exist are read, and a sample with nothing available is built without reading. A variable is unavailable only along dimensions it has: one without `time` is returned unchanged whatever is unavailable along `time`.

  The base class provides:
  - `locate(dim, labels)`: turns coordinate labels into positions. A scalar label gives a scalar position, which drops the dimension, or `None` if the source lacks the label. Several labels give an array, with `-1` for the ones the source lacks. It raises a `ValueError` if the source has repeated labels along the dimension, since such a label has no single position (pandas would return a slice or raise an unrelated error).
  - `labels(point)`: the coordinate labels of the selection, per dimension the view selects along and the source has an index for, without reading. By default the source's labels at the selected positions, with missing labels at missing positions (integer and boolean labels then become floats, and always when filling, as in `fetch`). A view overrides it to report labels it asks for although the source lacks them; `select` can then locate them. Where a position exists, an override must give the source's label there, which `fetch` checks: it calls an overridden `labels` once per fetch and raises a `ValueError` if the dimensions, shapes or labels disagree with `select`. The coordinates of what `fetch` returns are therefore always `labels(point)`, in the dtype of the coordinate. Without an override, `fetch` takes the labels from the positions it has and calls `select` only once.
  - `available(point)`: whether everything `select(point)` asks for exists, i.e. nothing would be filled. `accepts` is this for a rejecting view and always true for a filling one.
  - `accepts(point)`: whether the view can serve the point. Always true with `unavailable="fill"` (without calling `select`); otherwise it dry-runs `select` and checks that every returned position lies inside the source.
  - `fetch(point)`: reads the selection into memory, after the same check, so what it returns is loaded. Override it for views that are not a selection (data-dependent choices, computed values); an override has to return loaded data too.
- **`XrDataset`** (`base.py`): a `torch.utils.data.Dataset` that serves `transform(prepare(data))` for every point, `data` being `{view name: view.fetch(point)}`. A subclass overrides the methods `prepare` (the expensive part, which a cache stores) and `transform` (the cheap part); both return their input unchanged by default, so a plain `XrDataset` serves the fetched data. At construction it keeps only the points that every view accepts; their positions are in `indices`, a numpy array. `rejected="drop"` (the default) drops the rest, `rejected="raise"` reports them, and it always raises if no point survives. `validate=False` skips the check: every point is kept, `n_rejected` is `None`, and a point that a view would have rejected raises an `IndexError` when it is fetched. `n_rejected` otherwise holds the number of rejected points per view, each point counted for the first view that rejected it, and the error messages include it. `labels(name)` and `available(name)` collect what a view selects and whether it has all data for every sample, without reading: arrays with one row per sample, in the order of the samples.
- **`PointGrid`** (`points.py`): all combinations of the given coordinate labels, computed lazily, e.g. `PointGrid(time=times, lat=lats)`.
- **`PointTable`** (`points.py`): the rows of label arrays of equal length, e.g. `PointTable(time=times, lat=lats)` with `len(times)` points, for any set of points that is not a grid, such as the points of a grid that pass a filter, or those of a cache.
- **The layout of stores of samples** (`stores.py`, not re-exported from the package root): zarr 3 stores in which an array of samples has one chunk per sample and several samples per shard, a shard being one file of about 1 GB before compression by default (`SHARD_BYTES`). `new_group` creates the root of a store, `create_sample_array` and `new_sample_array` an array of samples, optionally with the names of its dimensions so that xarray can open it, and `SampleWriter` fills such an array in order, a whole shard at a time, holding at most one shard in memory. It came from sinfony_forecast_blending, which writes its caches, inference output and verification sums this way, and is the base of the cache.
- **The cache** (`XrDataset.cache`, `XrDataset.from_cache`, `cache.py`): A subclass whose `prepare` returns one `xr.Dataset` per sample, the prepared sample, can be cached. `dataset.cache(path, attrs, samples_per_shard, batch_size, num_workers, multiprocessing_context, overwrite)` writes the prepared sample of every point, through `write_cache` in `cache.py`: each data variable as an array with a leading `sample` dimension (variables of more than `SMALL_SAMPLE_BYTES`, 1 KiB, per sample with one chunk per sample in shards, smaller ones as one chunk), the coordinates once, the labels of the points as one array per dimension of the points, and `attrs` as the attributes of the store. A dataset whose class does not override `prepare` is not cached, unless it reads a cache itself. Which arrays are points and coordinates is recorded in the attribute `xarray_torch_dataset`. Every sample is checked against the first: the same data variables, dimensions, shapes and dtypes, and the same coordinates. `Subclass.from_cache(path, points, rejected)` returns an instance of the class it is called on, created without `__init__`, whose samples are `transform(prepared sample)`, read from the cache: all its points, or those given, of which those the cache lacks are rejected. Its `transform` must therefore not use what `__init__` sets, but may use `arguments`. Every dataset keeps the arguments of its constructor as `arguments`, by name and with defaults (recorded by `XrDataset.__new__`, for any subclass), and `cache` records `describe(arguments)` with the store (`arguments.py`): plain values in the attribute `xarray_torch_dataset`, arrays in full in the group `arguments/`, functions by name, other objects by class and attributes, xarray objects left out and listed. `from_cache(path, arguments=...)` describes the given arguments, with defaults, compares them with the record (`differences`), raises a `ValueError` naming the paths that differ, warns about what was left out and so not checked, and sets `arguments`; without them, `arguments` is the recorded description. A dict argument shaped like one of the record's markers would be misread. A dataset reads a cache, `cached`, when its only view is a `CacheView`. Underneath, `CacheView(path)` opens the store through xarray and serves the row of a point, found by its labels in a table built at construction, with the point's labels as coordinates; `CacheView.prepared_sample(row)` drops them again together with the attributes of the store, which gives back the prepared sample. `CacheView.points` is a `PointTable` of the cached points, `CacheView.attrs` the attributes given to `cache`.
- **Two views are included**, as examples of the two kinds of view and as what the tests and benchmarks run against. The package is meant to be imported into an application that writes its own views. They live in the `views/` subpackage, one module per kind, and are re-exported from `xarray_torch_dataset.views` and from the package root.
- **`OffsetView`** (`views/offset.py`): the declarative view. `offsets={"time": np.arange(-3, 1) * np.timedelta64(1, "h")}` selects the point's time and the three hours before it. Offsets are added to the point's label and the result is looked up with `locate`, so along time they are durations, not numbers of steps, and a label the source lacks is unavailable. Offsets along a time dimension must be timedeltas; integers raise a `TypeError`. Dimensions without offsets are selected at exactly the point's label and dropped from the result. Point entries the source has no index for are ignored. It overrides `labels` with the point's labels plus the offsets, and `select` locates those, so a filled entry shows the label that was asked for.
- **`FindLatestForecastView`** (`views/forecast.py`): the view with free-form selection logic. For forecast sources with `init_time` and `lead_time`. A point's init time and lead time give a valid time; among the source's init times within `init_time_offset_limits` of the point's init time, the view selects the most recent one that has a lead time reaching that valid time. This lets a coarser forecast (say, 3-hourly inits) be aligned with points on a finer grid.

- **Ready-made transforms** live in the `transforms/` subpackage, laid out like `views/`. A transform receives `{view name: xr.Dataset}` with the data already in memory, and is called from the `transform` method of a subclass. So far the subpackage holds **`ToTensors`** (`transforms/tensors.py`), an example like the views: it stacks each view's variables into one `float32` tensor, keeps only the values, and raises if the variables of a view have different dimensions. A dataset that overrides no `transform` returns the xarray data, which a `DataLoader` cannot batch.

The `examples/` folder has two scripts on random in-memory data. `examples/forecasts.py` aligns an hourly and a 3-hourly forecast with `FindLatestForecastView`, filling the points the coarser one cannot serve. `examples/history_windows.py` builds a dataset from three `OffsetView`s: the last four days, the same weekday in the two previous weeks, and the point's own day as the target. It prints:

```
9 of 16 points are kept
{'x': torch.Size([4, 2, 4, 8, 8]), 'x_weekly': torch.Size([4, 1, 2, 8, 8]), 'y': torch.Size([4, 1, 8, 8])}
```

Other facts:

- **Dependencies**: `numpy >= 2.1`, `torch >= 2.9`, `xarray >= 2025.7.0`, `zarr >= 3.1.0`. The package uses only long-established parts of them; the floors are the versions the test suite was last run against at its lowest (`uv pip install --resolution lowest-direct`, under Python 3.12 and 3.13), and they leave room for the pins of projects that use the package. zarr 3.1 is the first version that stores `datetime64` and `timedelta64` arrays in the zarr 3 format, which stores of samples need for their times; 3.0.8 failed on them. Dev dependencies: `ruff`, `mypy`, `pre-commit`, `pytest`, `dask` for some of the tests and the benchmarks, and `gcsfs` for the benchmarks. The package itself imports neither `dask` nor `gcsfs`.
- **Tooling**: pre-commit runs `ruff check --fix`, `ruff format`, `mypy` and `pytest` as local hooks through `uv run`, so tool versions come from `uv.lock`. Ruff lint rules are `E`, `F`, `I`, `UP`, `B`. mypy checks `src` only, with `check_untyped_defs` and `warn_unused_ignores` but not `strict`.
- **Tests**: a small pytest suite in `tests/` covers the package: the `View` contract (`locate`, labels and availability, accepting and rejecting, fetching, filling), the two included views, `ToTensors`, `PointGrid`, `XrDataset` and the layout of stores of samples. Run it with `uv run pytest`. It uses small in-memory data, and small stores in pytest's temporary directory, and runs as a pre-commit hook.
- **`README.md`** introduces the parts with a runnable example, shows how to write a view and says how to open zarr stores.
- **Benchmarks** live in `benchmarks/` and are run by hand against zarr stores on disk; see the next section.

## Zarr experiment (2026-10-10)

The package works on a zarr store without changes: samples are identical to in-memory ones, and validation does no I/O.

**Setup.** `benchmarks/make_stores.py` writes one synthetic year of hourly data (8760 x 128 x 128, two float32 variables, about 1 GB) as zarr v3 three times, with time chunks of 1, 24 and 168 steps. The dataset under test has an `OffsetView` with the last four hours of both variables, a single-step target and `ToTensors`. Two ways of opening a store are compared: "lazy" is `xr.open_dataset(path, engine="zarr", chunks=None)`, and "dask" is `xr.open_zarr(path)`. The machine has 8 cores. The data is random, so it does not compress, and the stores were in the OS page cache, so the numbers measure decoding and framework overhead, not disk or network.

```bash
uv run python benchmarks/make_stores.py STORES_DIR     # about 3 GB
uv run python benchmarks/per_sample.py STORES_DIR
uv run python benchmarks/loading.py STORES_DIR lazy,dask
uv run python benchmarks/fork_hang.py STORES_DIR/chunk24.zarr dask fork 1
```

**Forked workers can hang with dask.** If the parent process reads a sample from a dask-backed dataset and a `DataLoader` then starts forked workers (the default on Linux), the first batch never arrives. It reproduced in three of three attempts with `benchmarks/fork_hang.py`. The lazy backend is not affected. With dask, each of these avoids the hang: not reading in the parent first, the `spawn` or `forkserver` start method (about 8 s of worker start-up), or dask's synchronous scheduler (`DASK_SCHEDULER=synchronous`). The cause was not traced; see the design decisions for how it is handled.

**Per-sample cost** (`per_sample.py`), in ms per `dataset[i]` at random indices. The table and the two tables further down were measured on 2026-10-10 with an `OffsetView` that counted positions; the reads are the same with the current one, and a rerun of `per_sample.py` gave figures within 15% of this table. `loading.py` was not rerun in full. The figures in the list below are from that rerun (2026-10-11).

| Time chunk | In memory | Lazy | Dask |
|---|---|---|---|
| 1 | | 9.6 | 43.0 |
| 24 | 1.2 | 11.2 | 18.6 |
| 168 | | 45.5 | 44.4 |

- Framework overhead is about 1 ms per sample. For the history view on the 24-step store, `select` takes 0.12 ms and `fetch` 7.0 ms, against 6.0 ms for the same read through raw zarr.
- Dask is slower than lazy on small chunks and equal on large ones.
- Validation takes about 1.6 s for 8760 points and 2 views on every backend, which is roughly 90 µs per point and view.
- With `unavailable="fill"`, a sample that is entirely missing takes 4.6 ms against 6.9 ms for a complete one. It is no longer read (before that change it took 9.7 ms); what remains is xarray bookkeeping.

**DataLoader throughput** (`loading.py`), in samples per second, shuffled, batch size 32, `forkserver` workers. "One read" is batched reading, which was removed afterwards (see the design decisions); the rows are kept as the record of that experiment, and `loading.py` now measures only the per-sample rows.

| Time chunk | Backend | Reads | 0 workers | 4 workers | 8 workers |
|---|---|---|---|---|---|
| 1 | lazy | per sample | 88 | 310 | 601 |
| 1 | lazy | one read | 118 | 383 | 886 |
| 1 | dask | per sample | 22 | 77 | 147 |
| 1 | dask | one read | 55 | 172 | 444 |
| 24 | lazy | per sample | 85 | 293 | 479 |
| 24 | lazy | one read | 192 | 493 | 667 |
| 24 | dask | per sample | 54 | 180 | 271 |
| 24 | dask | one read | 120 | 326 | 471 |
| 168 | lazy | per sample | 21 | 84 | 146 |
| 168 | lazy | one read | 111 | 145 | 137 |
| 168 | dask | per sample | 22 | 83 | 113 |
| 168 | dask | one read | 78 | 132 | 148 |

- One read per batch helps most without workers (1.3 to 5.3 times the throughput) and less as workers are added. With 168-step chunks and 8 workers it makes no difference; why has not been investigated.
- The lazy backend is faster than dask on 1-step and 24-step chunks and about equal on 168-step chunks.

**Ways to read one batch**, in ms for the history view of 32 random points. This compared strategies for batched reading; `fetch_many` was the method that implemented it. The code for this table went with the feature and is in the history at commit `86880c7`.

| Time chunk | Backend | Per sample | One vectorized `isel` | Unique steps, then gather | `fetch_many` |
|---|---|---|---|---|---|
| 1 | lazy | 254 | 468 | 205 | 279 |
| 1 | dask | 1282 | 576 | 534 | 502 |
| 24 | lazy | 233 | 396 | 77 | 117 |
| 24 | dask | 367 | 159 | 127 | 160 |
| 168 | lazy | 985 | 446 | 150 | 196 |
| 168 | dask | 795 | 212 | 216 | 249 |

- "Unique steps, then gather" is a hand-written numpy version of what `fetch_many` did. `fetch_many` was 40 to 75 ms slower per batch on the lazy backend, which is the cost of building one `xr.Dataset` per point from the block. On the 1-step store that overhead cancels the gain for this view.
- Handing xarray one `(batch, time)` indexer takes 400 to 470 ms on the lazy backend whatever the chunking, so `fetch_many` did not use it. Why it is slow has not been investigated.

## WeatherBench 2 experiment (2026-10-11)

The first run on real data: IFS HRES forecasts paired with ERA5, both from the public WeatherBench 2 bucket on a 64 x 32 grid.

**Data.** The forecasts have 5114 init times (every 12 hours, 2016 to 2022) and 41 lead times (every 6 hours to 240 h), in chunks of 4 init times x 1 lead time x the whole field, compressed with Blosc/LZ4. ERA5 has 92,044 time steps (every 6 hours, 1959 to 2021) in chunks of 100 steps. `benchmarks/get_weatherbench2.py` copies three surface variables of the forecasts and the 2 m temperature of ERA5 to local disk byte for byte (5 GB, about 160,000 files, 13 minutes). Opening needs `decode_timedelta=True`, because lead times are stored as integer hours, and a rename of `time` / `prediction_timedelta` to `init_time` / `lead_time`. A local copy has to be opened with `consolidated=False`, since it holds only some of the variables.

**Dataset.** 35,088 points: two years of 6-hourly init times and 12 lead times (6 to 72 h). The forecasts are only 12-hourly, so `FindLatestForecastView` (three variables, window of -6 to 0 h) picks an earlier forecast with a longer lead time for every other point. The target is ERA5 at the valid time, through a three-line `ValidTimeView` defined in `benchmarks/weatherbench2.py`. A sample needs four chunks, one per forecast variable and one of ERA5.

```bash
uv run python benchmarks/get_weatherbench2.py ~/data/weatherbench2
uv run python benchmarks/weatherbench2.py ~/data/weatherbench2 cold
uv run python benchmarks/weatherbench2.py ~/data/weatherbench2
uv run python benchmarks/weatherbench2.py remote
```

**Results.** "Cold" evicts the local copy from the page cache before each measurement. Throughput is in samples per second from a shuffled DataLoader with batch size 32, as per-sample reading / batched reading. Batched reading was removed after this run; the script now measures per-sample reading only.

| Source | Construction | One sample | 0 workers | 4 workers | 8 workers |
|---|---|---|---|---|---|
| local, cold | 8.4 s | 11.8 ms | 142 / 159 | 545 / 547 | 773 / 887 |
| local, cached | 9.0 s | 10.4 ms | 162 / 181 | 554 / 604 | 797 / 892 |
| bucket | 10.8 s | 915 ms | 2 / 3 | 8 / 16 | 13 / 14 |

- 78 points are rejected, all by the target view, because their valid time lies after the end of ERA5: `n_rejected` is `{'forecast': 0, 'target': 78}`.
- Construction cost 120 to 150 µs per point and view in this run, more than with `OffsetView`, because the forecast view scanned all 5114 init times for every point. That was fixed afterwards: its `select` went from 146 to 25 µs, and construction from the local copy from 8.4 s to 3.6 s.
- A cold local disk makes little difference. This disk reads about 640 MB/s, and the chunks are small.
- Reading from the bucket is about 90 times slower per sample than from the local copy. The four chunks of a sample are requested one after another, at 200 to 300 ms each. Workers help in proportion, since the time is spent waiting. Each bucket figure covers only about 20 s, which is one to nine batches, so these are rough.
- Batched reading gained about 10% locally. `FindLatestForecastView` fell back to one read per point, so only the ERA5 target was read at once. Against the bucket it doubled throughput with 4 workers and made no difference with 8; given how few batches were timed, that is not a reliable difference.
- More workers than cores help against the bucket, since the time is spent waiting. A follow-up run of 60 s per setting gave 14 samples per second with 4 workers, 12 to 21 with 8 (three runs), 15 to 24 with 16 (two runs) and 55 with 32, on 8 cores. The spread between repeated runs is close to a factor of two, so only the trend is reliable. Even 55 is about 15 times below the local copy with 8 workers.

## Setting up on another machine

```bash
git clone git@gitlab.dkrz.de:b383481/xarray_torch_dataset.git
cd xarray_torch_dataset
uv sync                      # creates .venv from uv.lock
uv run pre-commit install    # the git hook is per clone, not part of the repo
uv run python examples/history_windows.py
```

Prerequisites that are per machine and not stored in the repo:

- `uv` (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Git identity (`git config user.name` / `user.email`).
- SSH access to the DKRZ GitLab (`gitlab.dkrz.de`), with a key registered in the GitLab account, since the repository is cloned over SSH.

The default `torch` wheel on Linux pulls in the CUDA libraries, so `.venv` is several GB even on a machine without a GPU.

## Design decisions so far

The first version had four abstract classes: `Anchor`, `Sampler`, `View` and `XrDataset`, with opaque points and views as free-form callables. It was reworked for these reasons:

- **`Anchor` and `Sampler` were removed.** They had the same interface (index to point), the framework only used the sampler, and choosing which indices to visit is already covered by torch's own samplers and `Subset`. What remains is the `Points` protocol. The name `Sampler` also collided with `torch.utils.data.Sampler`.
- **Points became concrete** (a dict of coordinate labels). With opaque points the framework could not validate, align or batch anything, and `XrDataset` was little more than a dict comprehension.
- **Views were split into `select` and `fetch`.** Selection logic stays unrestricted, but because it is separate from reading, the dataset can check up front which points a view can serve. This replaced the hand-matched `offset=history - 1` between sampler and view, which failed silently when wrong.
- **Views decide which points they accept, and the dataset enforces it.** A label missing in one source is now caught at construction instead of producing a wrong window at fetch time.
- **`select` returns positions, not labels** (`isel` instead of `sel`). The ready-made views compute positions anyway, dimensions without coordinates become selectable, and integer indexers are easier to combine for batched reads. The speed gain itself is small (roughly 50 to 130 µs per fetch on in-memory data). The price is that views must look labels up themselves, through `locate`, and that the base class has to guard against negative positions.
- **Missing data is marked by position, never by exception.** With `unavailable="fill"` a view has to say what shape the missing data would have had, and out-of-bounds positions carry that for free, including windows that are only partly missing. An exception carries no shape. An earlier version also accepted a `LookupError` from `select` when rejecting; that was removed so there is one mechanism, and so a `KeyError` from a typo is no longer mistaken for missing data.
- **A failed scalar lookup is `None`, not `-1`.** Arithmetic on `-1` gives positions that look valid (`-1 + [1, 2]` is `[0, 1]`), so a view that forgot to check would silently serve unrelated data. `None` raises a `TypeError` on that line instead. numpy's masked value and NaN were tried as markers that survive arithmetic and rejected: a masked value silently becomes position 0 when converted to a plain array, and NaN turns positions into floats. Array lookups still use `-1`, because an integer array cannot hold `None`; arithmetic on such an array keeps the old hazard.
- **`fetch` returns loaded data, and a transform never reads.** `fetch` used to return lazy data, so the read happened as a side effect inside the transform. That contradicted both names, and it was inconsistent: filled samples were already loaded. Laziness would only help to narrow a selection before reading, which in this design is what `select` is for. On dask-backed data it also meant that a transform touching a variable twice computed it twice.
- **A transform receives xarray objects, and `ToTensors` is only an example.** Handing over `{view name: xr.Dataset}` lets an application's transform use names, dimensions and coordinates. `ToTensors` keeps the values only. It used to pass variables of different dimensions to `to_array()`, which broadcast them silently (an orography was repeated along time); it now raises. A missing transform is not an error, because looking at the xarray samples directly is legitimate, but the docs say that a `DataLoader` needs one.
- **Filling reads only what exists and always returns floats.** Filling used to read position 0 as a stand-in for every missing position and blank it afterwards, so a sample with nothing available cost a full read, and a window at the end of a source read a chunk from its start. The stand-in is now a position the selection needs anyway, and when nothing exists along a dimension the data variables are built as NaN without reading. Separately, integer and boolean values became float only in samples where something was missing, so the dtype differed between samples; a filling view now returns them as floats for every sample.
- **A variable is unavailable only along dimensions it has.** Filling used to mask the whole dataset along each dimension with unavailable positions. A variable without that dimension, such as an orography next to a field over time, then gained the dimension and was blanked along it, so its shape differed between complete and filled samples and it lost data that was there. The mask is now applied only to the variables and coordinates that lie along the dimension, and the shortcut that skips reading applies only to those as well.
- **A dataset in which every view fills keeps every point**, including one that no source has data for, which becomes a sample of NaN. This is documented, not prevented: a rule such as "at least one view must have data" would make filling views run `select` during validation and add an option whose right setting depends on the application. Keeping one rejecting view, typically the target, is the answer.
- **One vocabulary for points a view cannot serve: views reject, the dataset handles what was rejected.** The view option is `unavailable="reject"` / `"fill"`, the method that asks a view is `accepts(point)`, and the dataset option is `rejected="drop"` / `"raise"`. These were `missing="drop"` / `"fill"`, `is_valid(point)` and `invalid="drop"` / `"raise"`. The old names had three problems. "Drop" was used at both levels, and wrongly for the view, which drops nothing: the dataset drops the point or raises. "Invalid" sounded like a separate condition, although a point was invalid exactly when a view could not serve it. And "missing" left open what is missing; the option is about data the source cannot supply for a complete point. It covers neither NaN values stored in the source nor a point that lacks a label the view needs, which is an error.
- **The included views are examples, and `OffsetView` works by label.** The package is a general-purpose tool; applications write their own views. Two views stay, one of each kind: `OffsetView` (declarative) and `FindLatestForecastView` (free-form logic). They are also what the tests and benchmarks run against. `OffsetView` used to count positions along the source's index, so on a time axis with gaps, or on a source of another resolution, a window silently covered the wrong period. It now adds the offsets to the point's label and looks the result up, which makes a label the source lacks unavailable instead. `SameWeekdayView` was removed, being this view with offsets of -14 and -7 days. Matching only exact labels is deliberate: a tolerance would bring back a quieter version of the same problem.
- **`FindLatestForecastView` searches sorted coordinates instead of scanning.** It used to compare every init time of the source with the point, which cost 146 µs per point on 5114 init times and grew with the archive. It now keeps sorted copies of the init and lead times and finds the candidates with `np.searchsorted`: the init times that lie in the window and from which some lead time can reach the valid time at all. The second condition bounds the candidates even without offset limits. `select` takes 20 to 27 µs whether the source has 500 or 50,000 init times. It no longer uses `locate` for the lead time, since the pandas lookup was most of the remaining cost, so it checks itself that both coordinates are free of repeated labels.
- **Batched reading was tried and removed.** From 2026-10-10 to 2026-10-11 the package could read a whole batch at once: `View.fetch_many(points)` read every position a batch needed in one block and split it per point, and `XrDataset(batched=True)` exposed that to a DataLoader through `__getitems__`. On synthetic windows from local zarr stores it gave 1.3 to 5.3 times the throughput without workers and between nothing and 3 times with eight. On real forecast data it gave about 10% more throughput, where eight workers gave about five times as much, because a view that needs scattered (init time, lead time) pairs cannot use a block read and fell back to one read per point. It also added a second method to the view contract, made one bad point fail a whole batch, and is unusual for torch datasets, where `__getitem__` with workers is the norm. The numbers are in the two experiment sections; the code is in the history up to commit `86880c7`.
- **Validation can be switched off, and then fails late but never silently.** With `validate=False` the dataset keeps every point without asking the views. `fetch` checks the positions again before reading, so a point that would have been rejected raises an `IndexError` at that moment instead of returning other data. The price is when it fails: possibly hours into training, inside a DataLoader worker. Views with `unavailable="fill"` behave the same either way. A view that overrides `fetch` has to do its own checking.
- **Rejected points are dropped by default, and counted.** Any dataset with a history window rejects a few points at the edges, so raising by default would fail on nearly every first use. To keep a misaligned source visible all the same, `XrDataset.n_rejected` records the number of rejected points per view, and the error messages include it. A point counts for the first view that rejected it: that is free, because the check stops there, whereas counting every rejecting view would mean asking all views about all points. There is no warning, since it would appear for every correct dataset.
- **`PointGrid` stays.** In one dimension a list comprehension does the same job, but point sets for forecasts have several dimensions (init time, lead time, possibly members or locations) and quickly reach hundreds of thousands of points. As a list of dicts that is a large object that is also copied into every DataLoader worker; `PointGrid` stores only the label arrays. A grid also knows that its dimensions are independent, which could later let validation run per dimension instead of per point. It cannot express arbitrary combinations of labels; for those, a list of dicts remains the way.
- **The dask fork hang is documented, not fixed.** It most likely comes from forking a process whose dask thread pool is already running, which this package cannot repair, and the cause was deliberately not traced. The recommendation, given in the README, is to open stores without dask, `xr.open_dataset(path, engine="zarr", chunks=None)`, which avoids the hang and in the benchmarks was faster on 1-step and 24-step chunks and about equal on 168-step chunks. For those who need dask, the `forkserver` start method or dask's synchronous scheduler work. There is no runtime warning for dask-backed sources, because it would also fire for these safe configurations.
- **A view reports the labels it asks for, beside the positions it selects.** An application that matches forecasts to points (sinfony_forecast_blending) records for every sample the run it chose and the lead time it needed, also when that run does not reach the valid time or no run lies in the window, and its model gets that lead time as input. Those labels are not in the source, so they could not be read from the fetched coordinates, which were NaT, and a spike needed an extra source of times for them. Selecting by labels instead of positions was not an option (see above). So `View.labels(point)` reports the labels, by default those at the selected positions, and a view that knows more overrides it; filling puts them on the coordinate of the dimension, so data and coordinates show what was asked for while the data stays NaN. Views that do not override it behave as before. An override is a second description of the same selection, so `fetch` checks that it agrees with `select` where the positions exist, which makes the coordinates of a fetched sample always equal `labels(point)`. Silently relabelling the data with `labels` would give the same guarantee, but would put wrong labels on real data whenever a view has a bug. The check costs one `labels` call per fetch, for overriding views only; it is not part of validation, so a disagreeing view fails at its first fetch. Two alternatives were tried or considered: documenting the rule only, which leaves a mistake unnoticed, and reindexing the source to an axis that contains every wanted label, so that positions alone carry the labels. On a zarr store opened without dask, `reindex` read the whole store into memory; with dask it stayed lazy, but a field took about 5 times as long and missing fields were read too. `View.available(point)` was added because a filling view accepts every point, and `XrDataset.labels`/`available` collect both for all samples, for example to write them next to cached samples.
- **The cache stores prepared samples, split off the transform.** An application's cache (sinfony_forecast_blending) holds neither what the views read nor the finished samples, but a state in between: the forecasts after reducing their members, which is the expensive part, and the times from which the model's inputs are computed when a sample is read, so that an experiment can shift a lead time on a cached dataset. Caching the views' data would keep the expensive part, caching the transformed samples would freeze the cheap part. So the transform is split in `prepare` and the rest, and the cache stores what `prepare` returns. Reading a cache is a view, `CacheView`, so a cached dataset is an `XrDataset` like any other, and a configuration that asks for part of a cache simply gives those points. The first version left the two stages to the application, with free functions `write_cache` and `CacheView` to wire together; an application then wrote the same lines each time, and caching was not visible on the dataset. So `cache` and `from_cache` became methods of `XrDataset`. The two stages were first arguments of the constructor, `prepare=` and `transform=`; they became methods that a subclass overrides, in the style of Lightning's modules, so that a dataset is written as one class, and `from_cache` can return an instance of that class, created without `__init__` since the constructor of a subclass opens sources that a cached dataset does not have. Checking that a cache belongs to its arguments was first left to the application, which compared settings it recorded itself. As every application would write that, the package records the arguments of the constructor itself, without the frame inspection of Lightning's `save_hyperparameters`: `XrDataset.__new__` receives the same arguments as `__init__`. Arrays are kept in full rather than as checksums, so that a cache shows what it was written from and an error can say what differs; the arguments are passed to `from_cache` as one dict, so that their names cannot clash with its own parameters. The layout of stores of samples came along from the same application, so that its other stores and the cache share it; zarr became a dependency for it. With DataLoader workers, `cache` takes a start method, as forking a process in which zarr runs threads gave a deadlock warning in the tests; `forkserver` avoids it.
- **Slices are not accepted from `select`.** No view used them, any slice can be written as `np.arange(...)`, and supporting them needed separate bounds and fill logic. They could return as a read optimisation for contiguous windows if a zarr test shows it matters.

## Open questions

None at the moment. What is left is listed under the known issues below.

## Known issues

Found by reading the code or by measuring; none has a fix yet.

- **Validation is slow on large point sets, and speeding it up is parked.** It is a Python loop over all points and views, at roughly 90 µs per point and view with `OffsetView`: 19 s for 105,000 points with two views, so about 3 minutes for a million. Looking labels up costs more than the position arithmetic of the earlier `OffsetView` (`select` went from 0.035 to 0.125 ms). `validate=False` avoids the cost for point sets that are known to be servable. The remedy would be an optional `accepts_many(points)` on `View`, with a default that loops, so that a view can look up the labels of many points in one call. It was not built because it adds to the view contract, each view needs its own vectorized version (real work for `FindLatestForecastView`), and it only pays off for point sets larger than any used so far. It can be added later without breaking existing views.

## Possible next steps

1. Check that the package installs into another project from the git URL, which is its purpose and has not been tried. A CI workflow that runs the pre-commit hooks would fit here.
2. Offer a supported way to reuse a validation. The WeatherBench 2 benchmark validates once and then assigns `indices` by hand on datasets built with `validate=False`; an application with a large point set will want the same across runs.
3. Use the package in an application; what turns up there should set the priorities. sinfony_forecast_blending is moving its combined dataset onto the package; the selected labels, the layout of stores of samples and the cache came from that.
4. Speed up validation (see the known issues) once point sets are large enough to make construction time a nuisance.
5. Only if remote stores have to be read directly: read the variables of a sample concurrently, since per-sample time there is the sum of sequential requests (four of 200 to 300 ms each in the WeatherBench 2 run). Copying the data to local disk made reads about 90 times faster, and many workers help as well, so this is not planned.
