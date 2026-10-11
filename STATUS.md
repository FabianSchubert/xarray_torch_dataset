# Project status

Snapshot as of 2026-10-11. This file records where the project stands and which design questions are open, so work can continue on another machine.

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
  - `unavailable="fill"`: the view accepts every point, and `fetch` returns data of the full expected shape with NaN at the missing positions (so a window can be partly filled). Coordinates of missing entries are masked too (NaN or NaT). Integer and boolean variables and coordinates come back as floats, for complete samples as well, so the dtype does not depend on the sample. Only positions that exist are read, and a sample with nothing available is built without reading. A variable is unavailable only along dimensions it has: one without `time` is returned unchanged whatever is unavailable along `time`.

  The base class provides:
  - `locate(dim, labels)`: turns coordinate labels into positions. A scalar label gives a scalar position, which drops the dimension, or `None` if the source lacks the label. Several labels give an array, with `-1` for the ones the source lacks. It raises a `ValueError` if the source has repeated labels along the dimension, since such a label has no single position (pandas would return a slice or raise an unrelated error).
  - `accepts(point)`: whether the view can serve the point. Always true with `unavailable="fill"` (without calling `select`); otherwise it dry-runs `select` and checks that every returned position lies inside the source.
  - `fetch(point)`: reads the selection into memory, after the same check, so what it returns is loaded. Override it for views that are not a selection (data-dependent choices, computed values); an override has to return loaded data too.
  - `fetch_many(points)`: what `fetch` gives for each point, as a list, from a single read. It reads every position that some point needs once, loads that block into memory, and takes each point's selection from it, so points may have different shapes. It falls back to one `fetch` per point in two cases: when the view overrides `fetch`, and when the block would hold more elements than the points ask for. The second case arises because along several dimensions the block is an outer product; it applies to `FindLatestForecastView`, where each point needs one scattered (init time, lead time) pair.
- **`XrDataset`** (`base.py`): a `torch.utils.data.Dataset`. At construction it keeps only the points that every view accepts; their positions are in `indices`, a numpy array. `rejected="drop"` (the default) drops the rest, `rejected="raise"` reports them, and it always raises if no point survives. `validate=False` skips the check: every point is kept, `n_rejected` is `None`, and a point that a view would have rejected raises an `IndexError` when it is fetched. `n_rejected` otherwise holds the number of rejected points per view, each point counted for the first view that rejected it, and the error messages include it. By default a `DataLoader` reads it sample by sample through `__getitem__`. With `batched=True`, which is experimental, it exposes `__getitems__`, which a batching `DataLoader` calls with all indices of a batch; that goes through `fetch_many`, so a batch costs one read per view instead of one per sample.
- **`PointGrid`** (`points.py`): all combinations of the given coordinate labels, computed lazily, e.g. `PointGrid(time=times, lat=lats)`.
- **Two views are included**, as examples of the two kinds of view and as what the tests and benchmarks run against. The package is meant to be imported into an application that writes its own views. They live in the `views/` subpackage, one module per kind, and are re-exported from `xarray_torch_dataset.views` and from the package root.
- **`OffsetView`** (`views/offset.py`): the declarative view. `offsets={"time": np.arange(-3, 1) * np.timedelta64(1, "h")}` selects the point's time and the three hours before it. Offsets are added to the point's label and the result is looked up with `locate`, so along time they are durations, not numbers of steps, and a label the source lacks is unavailable. Offsets along a time dimension must be timedeltas; integers raise a `TypeError`. Dimensions without offsets are selected at exactly the point's label and dropped from the result. Point entries the source has no index for are ignored.
- **`FindLatestForecastView`** (`views/forecast.py`): the view with free-form selection logic. For forecast sources with `init_time` and `lead_time`. A point's init time and lead time give a valid time; among the source's init times within `init_time_offset_limits` of the point's init time, the view selects the most recent one that has a lead time reaching that valid time. This lets a coarser forecast (say, 3-hourly inits) be aligned with points on a finer grid.

- **Ready-made transforms** live in the `transforms/` subpackage, laid out like `views/`. A transform receives `{view name: xr.Dataset}` with the data already in memory. So far the subpackage holds **`ToTensors`** (`transforms/tensors.py`), an example like the views: it stacks each view's variables into one `float32` tensor, keeps only the values, and raises if the variables of a view have different dimensions. Without a transform a dataset returns the xarray data, which a `DataLoader` cannot batch.

The `examples/` folder has two scripts on random in-memory data. `examples/forecasts.py` aligns an hourly and a 3-hourly forecast with `FindLatestForecastView`, filling the points the coarser one cannot serve. `examples/history_windows.py` builds a dataset from three `OffsetView`s: the last four days, the same weekday in the two previous weeks, and the point's own day as the target. It prints:

```
9 of 16 points are kept
{'x': torch.Size([4, 2, 4, 8, 8]), 'x_weekly': torch.Size([4, 1, 2, 8, 8]), 'y': torch.Size([4, 1, 8, 8])}
```

Other facts:

- **Dependencies**: `numpy`, `torch`, `xarray`. Dev dependencies: `ruff`, `mypy`, `pre-commit`, `pytest`, and `zarr` and `dask` for the benchmarks. The package itself imports neither.
- **Tooling**: pre-commit runs `ruff check --fix`, `ruff format`, `mypy` and `pytest` as local hooks through `uv run`, so tool versions come from `uv.lock`. Ruff lint rules are `E`, `F`, `I`, `UP`, `B`. mypy checks `src` only, with `check_untyped_defs` and `warn_unused_ignores` but not `strict`.
- **Tests**: a small pytest suite in `tests/` covers the package: the `View` contract (`locate`, accepting and rejecting, fetching, filling), `fetch_many` against `fetch`, the two included views, `ToTensors`, `PointGrid` and `XrDataset`. Run it with `uv run pytest`. It uses small in-memory data only and runs as a pre-commit hook.
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

**DataLoader throughput** (`loading.py`), in samples per second, shuffled, batch size 32, `forkserver` workers. "Per sample" is the default; "one read" is `batched=True`.

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

**Ways to read one batch** (`loading.py`), in ms for the history view of 32 random points:

| Time chunk | Backend | Per sample | One vectorized `isel` | Unique steps, then gather | `fetch_many` |
|---|---|---|---|---|---|
| 1 | lazy | 254 | 468 | 205 | 279 |
| 1 | dask | 1282 | 576 | 534 | 502 |
| 24 | lazy | 233 | 396 | 77 | 117 |
| 24 | dask | 367 | 159 | 127 | 160 |
| 168 | lazy | 985 | 446 | 150 | 196 |
| 168 | dask | 795 | 212 | 216 | 249 |

- "Unique steps, then gather" is a hand-written numpy version of what `fetch_many` does. `fetch_many` is 40 to 75 ms slower per batch on the lazy backend, which is the cost of building one `xr.Dataset` per point from the block. On the 1-step store that overhead cancels the gain for this view.
- Handing xarray one `(batch, time)` indexer takes 400 to 470 ms on the lazy backend whatever the chunking, so `fetch_many` does not use it. Why it is slow has not been investigated.

## Setting up on another machine

```bash
git clone https://github.com/FabianSchubert/xarray_torch_dataset.git
cd xarray_torch_dataset
uv sync                      # creates .venv from uv.lock
uv run pre-commit install    # the git hook is per clone, not part of the repo
uv run python examples/history_windows.py
```

Prerequisites that are per machine and not stored in the repo:

- `uv` (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Git identity (`git config user.name` / `user.email`).
- GitHub authentication for the private repo. With the GitHub CLI: `gh auth login`, then `gh auth setup-git` so plain `git pull` / `git push` work over HTTPS.

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
- **`fetch` returns loaded data, and a transform never reads.** `fetch` used to return lazy data, so the read happened as a side effect inside the transform. That contradicted both names, and it was inconsistent: filled samples and batched reads were already loaded. Laziness would only help to narrow a selection before reading, which in this design is what `select` is for. On dask-backed data it also meant that a transform touching a variable twice computed it twice.
- **A transform receives xarray objects, and `ToTensors` is only an example.** Handing over `{view name: xr.Dataset}` lets an application's transform use names, dimensions and coordinates. `ToTensors` keeps the values only. It used to pass variables of different dimensions to `to_array()`, which broadcast them silently (an orography was repeated along time); it now raises. A missing transform is not an error, because looking at the xarray samples directly is legitimate, but the docs say that a `DataLoader` needs one.
- **Filling reads only what exists and always returns floats.** Filling used to read position 0 as a stand-in for every missing position and blank it afterwards, so a sample with nothing available cost a full read, and a window at the end of a source read a chunk from its start. The stand-in is now a position the selection needs anyway, and when nothing exists along a dimension the data variables are built as NaN without reading. Separately, integer and boolean values became float only in samples where something was missing, so the dtype differed between samples; a filling view now returns them as floats for every sample.
- **A variable is unavailable only along dimensions it has.** Filling used to mask the whole dataset along each dimension with unavailable positions. A variable without that dimension, such as an orography next to a field over time, then gained the dimension and was blanked along it, so its shape differed between complete and filled samples and it lost data that was there. The mask is now applied only to the variables and coordinates that lie along the dimension, and the shortcut that skips reading applies only to those as well.
- **A dataset in which every view fills keeps every point**, including one that no source has data for, which becomes a sample of NaN. This is documented, not prevented: a rule such as "at least one view must have data" would make filling views run `select` during validation and add an option whose right setting depends on the application. Keeping one rejecting view, typically the target, is the answer.
- **Batched reading is opt-in.** It was added on 2026-10-10 as always on. It is now behind `batched=True`, because reading a batch at once is not what users of a torch dataset expect, it changes failure behaviour (one bad point fails a batch), and its benefit is unproven on real data. `__getitems__` is a property that is `None` unless `batched=True`, which is how torch's DataLoader and `Subset` decide whether to use it.
- **A batch is read as one block, then split per point.** `fetch_many` returns a list with one dataset per point instead of one stacked dataset. Points with different shapes therefore need no special handling, a view's contract stays "one dataset per point", and stacking stays with torch's collate function. The price is the per-point overhead measured above.
- **One vocabulary for points a view cannot serve: views reject, the dataset handles what was rejected.** The view option is `unavailable="reject"` / `"fill"`, the method that asks a view is `accepts(point)`, and the dataset option is `rejected="drop"` / `"raise"`. These were `missing="drop"` / `"fill"`, `is_valid(point)` and `invalid="drop"` / `"raise"`. The old names had three problems. "Drop" was used at both levels, and wrongly for the view, which drops nothing: the dataset drops the point or raises. "Invalid" sounded like a separate condition, although a point was invalid exactly when a view could not serve it. And "missing" left open what is missing; the option is about data the source cannot supply for a complete point. It covers neither NaN values stored in the source nor a point that lacks a label the view needs, which is an error.
- **The included views are examples, and `OffsetView` works by label.** The package is a general-purpose tool; applications write their own views. Two views stay, one of each kind: `OffsetView` (declarative) and `FindLatestForecastView` (free-form logic). They are also what the tests and benchmarks run against. `OffsetView` used to count positions along the source's index, so on a time axis with gaps, or on a source of another resolution, a window silently covered the wrong period. It now adds the offsets to the point's label and looks the result up, which makes a label the source lacks unavailable instead. `SameWeekdayView` was removed, being this view with offsets of -14 and -7 days. Matching only exact labels is deliberate: a tolerance would bring back a quieter version of the same problem.
- **Validation can be switched off, and then fails late but never silently.** With `validate=False` the dataset keeps every point without asking the views. `fetch` checks the positions again before reading, so a point that would have been rejected raises an `IndexError` at that moment instead of returning other data. The price is when it fails: possibly hours into training, inside a DataLoader worker, and for a whole batch at once when batches are read through `fetch_many`. Views with `unavailable="fill"` behave the same either way. A view that overrides `fetch` has to do its own checking.
- **Rejected points are dropped by default, and counted.** Any dataset with a history window rejects a few points at the edges, so raising by default would fail on nearly every first use. To keep a misaligned source visible all the same, `XrDataset.n_rejected` records the number of rejected points per view, and the error messages include it. A point counts for the first view that rejected it: that is free, because the check stops there, whereas counting every rejecting view would mean asking all views about all points. There is no warning, since it would appear for every correct dataset.
- **`PointGrid` stays.** In one dimension a list comprehension does the same job, but point sets for forecasts have several dimensions (init time, lead time, possibly members or locations) and quickly reach hundreds of thousands of points. As a list of dicts that is a large object that is also copied into every DataLoader worker; `PointGrid` stores only the label arrays. A grid also knows that its dimensions are independent, which could later let validation run per dimension instead of per point. It cannot express arbitrary combinations of labels; for those, a list of dicts remains the way.
- **The dask fork hang is documented, not fixed.** It most likely comes from forking a process whose dask thread pool is already running, which this package cannot repair, and the cause was deliberately not traced. The recommendation, given in the README, is to open stores without dask, `xr.open_dataset(path, engine="zarr", chunks=None)`, which avoids the hang and in the benchmarks was faster on 1-step and 24-step chunks and about equal on 168-step chunks. For those who need dask, the `forkserver` start method or dask's synchronous scheduler work. There is no runtime warning for dask-backed sources, because it would also fire for these safe configurations.
- **Slices are not accepted from `select`.** No view used them, any slice can be written as `np.arange(...)`, and supporting them needed separate bounds and fill logic. They could return as a read optimisation for contiguous windows if a zarr test shows it matters.

## Open questions

None at the moment. What is left is listed under the known issues and the experimental feature below.

## Known issues

Found by reading the code or by measuring; none has a fix yet.

- **Validation is slow on large point sets, and speeding it up is parked.** It is a Python loop over all points and views, at roughly 90 µs per point and view with `OffsetView`: 19 s for 105,000 points with two views, so about 3 minutes for a million. Looking labels up costs more than the position arithmetic of the earlier `OffsetView` (`select` went from 0.035 to 0.125 ms). `validate=False` avoids the cost for point sets that are known to be servable. The remedy would be an optional `accepts_many(points)` on `View`, with a default that loops, so that a view can look up the labels of many points in one call. It was not built because it adds to the view contract, each view needs its own vectorized version (real work for `FindLatestForecastView`), and it only pays off for point sets larger than any used so far. It can be added later without breaking existing views.
- **`FindLatestForecastView.select` scans every init time for every point.** That adds to the validation cost above on long forecast archives.

## Experimental: batched reading

`XrDataset(..., batched=True)` and `View.fetch_many` work and are tested, but are off by default and not being developed further for now. Reading a batch at once is unusual for torch datasets; the usual `__getitem__` with DataLoader workers is the default here too. The gain was measured only on random data in the page cache (see the zarr experiment above), where it shrank as workers were added. What is known to be unfinished:

- Views that need scattered elements along several dimensions, such as `FindLatestForecastView`, fall back to one read per point and gain nothing. A pointwise read would serve them, but xarray's vectorized indexing was slow in the test above.
- The fallback rule counts elements, not chunks. Reading a somewhat larger block can still be cheaper when it touches the same chunks.
- `fetch_many` has 40 to 75 ms of overhead per batch of 32 from building one dataset per point. It could be trimmed, for example by letting the transform work on the block.
- One point that cannot be fetched fails its whole batch.

Whether to finish or remove it should be decided once there are numbers from real data.

## Possible next steps

1. Repeat the benchmarks on a store that is not in the page cache, ideally real, compressible data on network storage, since that is where fewer reads should matter most.
2. With those numbers, decide whether to finish or remove batched reading.
