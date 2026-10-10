# Project status

Snapshot as of 2026-10-10. This file records where the project stands and which design questions are open, so work can continue on another machine.

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
- **`View`** (`base.py`): abstract. Subclasses implement `select(point)`, which returns positional indexers as accepted by `xr.Dataset.isel` and does no I/O; it may contain arbitrary logic. Positions are integers or integer arrays; slices are not supported. Missing data is marked by position, and that is the only way to mark it: `None` for a single position, and a position outside the source (by convention `-1`) for an entry of an array. Negative positions never count from the end, and any exception from `select` is treated as a bug. What happens to a point with missing data is set per view by `missing`:
  - `missing="drop"` (the default): the point is invalid for this view.
  - `missing="fill"`: the point is always valid, and `fetch` returns data of the full expected shape with NaN at the missing positions (so a window can be partly filled). Coordinates of missing entries are masked too (NaN or NaT).

  The base class provides:
  - `locate(dim, labels)`: turns coordinate labels into positions. A scalar label gives a scalar position, which drops the dimension, or `None` if the source lacks the label. Several labels give an array, with `-1` for the ones the source lacks.
  - `is_valid(point)`: always true with `missing="fill"` (without calling `select`); otherwise it dry-runs `select` and checks that every returned position lies inside the source.
  - `fetch(point)`: reads the selection, after the same check. Override it for views that are not a selection (data-dependent choices, computed values).
  - `fetch_many(points)`: what `fetch` gives for each point, as a list, from a single read. It reads every position that some point needs once, loads that block into memory, and takes each point's selection from it, so points may have different shapes. It falls back to one `fetch` per point in two cases: when the view overrides `fetch`, and when the block would hold more elements than the points ask for. The second case arises because along several dimensions the block is an outer product; it applies to `FindLatestForecastView`, where each point needs one scattered (init time, lead time) pair.
- **`XrDataset`** (`base.py`): a `torch.utils.data.Dataset`. At construction it keeps only the points for which every view is valid. `invalid="drop"` (the default) drops the rest silently, `invalid="raise"` reports them, and it always raises if no point survives. It also implements `__getitems__`, which a batching `DataLoader` calls with all indices of a batch; that goes through `fetch_many`, so a batch costs one read per view instead of one per sample.
- **`PointGrid`** (`points.py`): all combinations of the given coordinate labels, computed lazily, e.g. `PointGrid(time=times, lat=lats)`.
- **Ready-made views** live in the `views/` subpackage, one module per kind, and are re-exported from `xarray_torch_dataset.views` and from the package root.
- **`OffsetView`** (`views/offset.py`): the declarative view. `offsets={"time": [-3, -2, -1, 0]}` selects positional steps relative to the point, contiguous or not. Dimensions without offsets are selected at exactly the point's label and dropped from the result. Point entries the source has no index for are ignored. It accepts `missing` like every view; with `"fill"`, steps that leave the source come back as NaN.
- **`SameWeekdayView`** (`views/weekday.py`): the same weekday in each of the `weeks` weeks before the point, along `time`. It counts durations rather than steps, so it shows free-form selection logic through `locate`.
- **`FindLatestForecastView`** (`views/forecast.py`): for forecast sources with `init_time` and `lead_time`. A point's init time and lead time give a valid time; among the source's init times within `init_time_offset_limits` of the point's init time, the view selects the most recent one that has a lead time reaching that valid time. This lets a coarser forecast (say, 3-hourly inits) be aligned with points on a finer grid.

- **Ready-made transforms** live in the `transforms/` subpackage, laid out like `views/`. So far it holds **`ToTensors`** (`transforms/tensors.py`), which stacks each view's variables into one float tensor.

The `examples/` folder has two scripts on random in-memory data. `examples/forecasts.py` aligns an hourly and a 3-hourly forecast with `FindLatestForecastView`, filling the points the coarser one cannot serve. `examples/history_windows.py` builds a dataset from two `OffsetView`s (a four-step history and a target) and a `SameWeekdayView` (the same weekday in the two previous weeks). It prints:

```
9 of 16 points are valid
{'x': torch.Size([4, 2, 4, 8, 8]), 'x_weekly': torch.Size([4, 1, 2, 8, 8]), 'y': torch.Size([4, 1, 8, 8])}
```

Other facts:

- **Dependencies**: `numpy`, `torch`, `xarray`, `matplotlib`. Dev dependencies: `ruff`, `mypy`, `pre-commit`, `pytest`, and `zarr` and `dask` for the benchmarks. The package itself imports neither.
- **Tooling**: pre-commit runs `ruff check --fix`, `ruff format`, `mypy` and `pytest` as local hooks through `uv run`, so tool versions come from `uv.lock`. Ruff lint rules are `E`, `F`, `I`, `UP`, `B`. mypy checks `src` only, with `check_untyped_defs` and `warn_unused_ignores` but not `strict`.
- **Tests**: a small pytest suite in `tests/` covers the package: the `View` contract (`locate`, validity, fetching, filling), `fetch_many` against `fetch`, the three ready-made views, `ToTensors`, `PointGrid` and `XrDataset`. Run it with `uv run pytest`. It uses small in-memory data only and runs as a pre-commit hook.
- **Benchmarks** live in `benchmarks/` and are run by hand against zarr stores on disk; see the next section.

## Zarr experiment (2026-10-10)

The package works on a zarr store without changes: samples are identical to in-memory ones, and validation does no I/O.

**Setup.** `benchmarks/make_stores.py` writes one synthetic year of hourly data (8760 x 128 x 128, two float32 variables, about 1 GB) as zarr v3 three times, with time chunks of 1, 24 and 168 steps. The dataset under test has a four-step `OffsetView` history on both variables, a single-step target and `ToTensors`. Two ways of opening a store are compared: "lazy" is `xr.open_dataset(path, engine="zarr", chunks=None)`, and "dask" is `xr.open_zarr(path)`. The machine has 8 cores. The data is random, so it does not compress, and the stores were in the OS page cache, so the numbers measure decoding and framework overhead, not disk or network.

```bash
uv run python benchmarks/make_stores.py STORES_DIR     # about 3 GB
uv run python benchmarks/per_sample.py STORES_DIR
uv run python benchmarks/loading.py STORES_DIR lazy,dask
uv run python benchmarks/fork_hang.py STORES_DIR/chunk24.zarr dask fork 1
```

**Forked workers can hang with dask.** If the parent process reads a sample from a dask-backed dataset and a `DataLoader` then starts forked workers (the default on Linux), the first batch never arrives. It reproduced in three of three attempts with `benchmarks/fork_hang.py`. The lazy backend is not affected. With dask, each of these avoids the hang: not reading in the parent first, the `spawn` or `forkserver` start method (about 8 s of worker start-up), or dask's synchronous scheduler (`DASK_SCHEDULER=synchronous`). The cause has not been investigated further.

**Per-sample cost** (`per_sample.py`), in ms per `dataset[i]` at random indices:

| Time chunk | In memory | Lazy | Dask |
|---|---|---|---|
| 1 | | 9.6 | 43.0 |
| 24 | 1.2 | 11.2 | 18.6 |
| 168 | | 45.5 | 44.4 |

- Framework overhead is about 1 ms per sample. For the history view on the 24-step store, `select` takes 0.035 ms and fetch plus load 7.3 ms, against 6.7 ms for the same read through raw zarr.
- Dask is slower than lazy on small chunks and equal on large ones.
- Validation takes about 0.8 s for 8760 points and 2 views on every backend, which is roughly 45 µs per point and view.
- With `missing="fill"`, a sample that is entirely missing takes 9.4 ms against 7.3 ms for a complete one.

**DataLoader throughput** (`loading.py`), in samples per second, shuffled, batch size 32, `forkserver` workers. "Per sample" disables `__getitems__`; "one read" is the default.

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
- **Validity is owned by the views and enforced by the dataset.** A label missing in one source is now caught at construction instead of producing a wrong window at fetch time.
- **`select` returns positions, not labels** (`isel` instead of `sel`). The ready-made views compute positions anyway, dimensions without coordinates become selectable, and integer indexers are easier to combine for batched reads. The speed gain itself is small (roughly 50 to 130 µs per fetch on in-memory data). The price is that views must look labels up themselves, through `locate`, and that the base class has to guard against negative positions.
- **Missing data is marked by position, never by exception.** With `missing="fill"` a view has to say what shape the missing data would have had, and out-of-bounds positions carry that for free, including windows that are only partly missing. An exception carries no shape. An earlier version also accepted a `LookupError` from `select` in drop mode; that was removed so there is one mechanism, and so a `KeyError` from a typo is no longer mistaken for missing data.
- **A failed scalar lookup is `None`, not `-1`.** Arithmetic on `-1` gives positions that look valid (`-1 + [1, 2]` is `[0, 1]`), so a view that forgot to check would silently serve unrelated data. `None` raises a `TypeError` on that line instead. numpy's masked value and NaN were tried as markers that survive arithmetic and rejected: a masked value silently becomes position 0 when converted to a plain array, and NaN turns positions into floats. Array lookups still use `-1`, because an integer array cannot hold `None`; arithmetic on such an array keeps the old hazard.
- **A batch is read as one block, then split per point.** `fetch_many` returns a list with one dataset per point instead of one stacked dataset. Points with different shapes therefore need no special handling, a view's contract stays "one dataset per point", and stacking stays with torch's collate function. The price is the per-point overhead measured above.
- **Slices are not accepted from `select`.** No view used them, any slice can be written as `np.arange(...)`, and supporting them needed separate bounds and fill logic. They could return as a read optimisation for contiguous windows if a zarr test shows it matters.

## Open questions

1. **Is `PointGrid` worth having?** For one dimension a list comprehension does the same job; it only earns its place for multi-dimensional grids.
2. **Default for invalid points.** Dropping suits the history-margin case but can hide a misaligned source. The alternative is to default to `invalid="raise"`.
3. **Steps versus durations.** `OffsetView` counts positional steps along the source's index. On a time axis with gaps, or across sources with different resolutions, a step window covers the wrong period without any error. A duration-based view (or option) is not designed yet.
4. **Validation cost.** Validation is a Python loop over all points and views, at roughly 45 µs per point and view. Extrapolated, that is about 90 s for a million points with two views; it has not been run at that size.
5. **Batched fetching** is prototyped (`fetch_many`, `__getitems__`) but not settled:
   - Views that need scattered elements along several dimensions, such as `FindLatestForecastView`, fall back to one read per point. A pointwise read would serve them, but xarray's vectorized indexing was slow in the test above.
   - The fallback rule counts elements, not chunks. Reading a somewhat larger block can still be cheaper when it touches the same chunks.
   - The per-point overhead of `fetch_many` could be trimmed, for example by letting the transform work on the block.
   - `__getitems__` is always on. There is no switch to read sample by sample, short of setting `__getitems__ = None` on a subclass as the benchmark does.
6. **Rough edges of `missing="fill"`.** A filled sample still reads real data (position 0 stands in for each missing position) before masking it, which costs I/O on lazy sources; on zarr an entirely missing sample costs as much as a complete one. Integer variables become float in filled samples only, so dtypes can differ between samples. If every view fills, no point is ever dropped, even one that is NaN in all sources; a rule like "at least one view must have data" would belong in `XrDataset`.
7. **What a transform receives.** `fetch` returns an `xr.Dataset`, and `ToTensors` assumes all variables in a view share the same dims because it calls `to_array()`. That is also where lazy data actually gets loaded.
8. **Dask and forked workers.** The hang described above is easy to run into. Options are to document it, to recommend the lazy backend (which was also faster here), or to detect a dask-backed source and warn.

## Possible next steps

1. Decide open questions 1 and 2; they are small and only need a choice.
2. Decide how to handle the dask fork hang (question 8).
3. Repeat the benchmarks on a store that is not in the page cache, ideally real, compressible data on network storage, since that is where fewer reads should matter most.
4. Settle batched fetching (question 5), starting with the forecast view.
5. Design duration-based windows (question 3).
