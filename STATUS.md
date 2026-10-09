# Project status

Snapshot as of 2026-10-09. This file records where the project stands and which design questions are open, so work can continue on another machine.

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
- **`XrDataset`** (`base.py`): a `torch.utils.data.Dataset`. At construction it keeps only the points for which every view is valid. `invalid="drop"` (the default) drops the rest silently, `invalid="raise"` reports them, and it always raises if no point survives.
- **`PointGrid`** (`points.py`): all combinations of the given coordinate labels, computed lazily, e.g. `PointGrid(time=times, lat=lats)`.
- **`OffsetView`** (`views.py`): the declarative view. `offsets={"time": [-3, -2, -1, 0]}` selects positional steps relative to the point, contiguous or not. Dimensions without offsets are selected at exactly the point's label and dropped from the result. Point entries the source has no index for are ignored. It accepts `missing` like every view; with `"fill"`, steps that leave the source come back as NaN.

`example.py` builds a dataset on random in-memory data from two `OffsetView`s (a four-step history and a target) and a custom `SameWeekdayView` (the same weekday in the two previous weeks) that shows free-form selection logic. It prints:

```
9 of 16 points are valid
{'x': torch.Size([4, 2, 4, 8, 8]), 'x_weekly': torch.Size([4, 1, 2, 8, 8]), 'y': torch.Size([4, 1, 8, 8])}
```

Other facts:

- **Dependencies**: `numpy`, `torch`, `xarray`. Dev dependencies: `ruff`, `mypy`, `pre-commit`.
- **Tooling**: pre-commit runs `ruff check --fix`, `ruff format`, `mypy` and `pytest` as local hooks through `uv run`, so tool versions come from `uv.lock`. Ruff lint rules are `E`, `F`, `I`, `UP`, `B`. mypy checks `src` only, with `check_untyped_defs` and `warn_unused_ignores` but not `strict`.
- **Tests**: a small pytest suite in `tests/` covers the package: the `View` contract (`locate`, validity, fetching, filling), `OffsetView`, `PointGrid` and `XrDataset`. Run it with `uv run pytest`. It uses small in-memory data only and runs as a pre-commit hook. The views in `examples/` are not covered.
- The code has only been exercised on small in-memory data, never on zarr or dask-backed sources.

## Setting up on another machine

```bash
git clone https://github.com/FabianSchubert/xarray_torch_dataset.git
cd xarray_torch_dataset
uv sync                      # creates .venv from uv.lock
uv run pre-commit install    # the git hook is per clone, not part of the repo
uv run python example.py
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
- **Slices are not accepted from `select`.** No view used them, any slice can be written as `np.arange(...)`, and supporting them needed separate bounds and fill logic. They could return as a read optimisation for contiguous windows if a zarr test shows it matters.

## Open questions

1. **Is `PointGrid` worth having?** For one dimension a list comprehension does the same job; it only earns its place for multi-dimensional grids.
2. **Default for invalid points.** Dropping suits the history-margin case but can hide a misaligned source. The alternative is to default to `invalid="raise"`.
3. **Steps versus durations.** `OffsetView` counts positional steps along the source's index. On a time axis with gaps, or across sources with different resolutions, a step window covers the wrong period without any error. A duration-based view (or option) is not designed yet.
4. **Validation cost.** Validation is a Python loop over all points and views. It is fine at example size and has not been timed on millions of points.
5. **Batched fetching.** Each `__getitem__` does one read per view. On zarr or dask-backed data with DataLoader workers that will likely be the bottleneck. The `select` split makes it possible to combine the indexers of a batch into one read per source (torch supports this through `__getitems__`), but combining gets awkward when points yield different shapes. This needs a prototype against a real zarr store before the contract is settled.
6. **Rough edges of `missing="fill"`.** A filled sample still reads real data (position 0 stands in for each missing position) before masking it, which costs I/O on lazy sources. Integer variables become float in filled samples only, so dtypes can differ between samples. If every view fills, no point is ever dropped, even one that is NaN in all sources; a rule like "at least one view must have data" would belong in `XrDataset`.
7. **What a transform receives.** `fetch` returns an `xr.Dataset`, and the example's `ToTensors` assumes all variables in a view share the same dims because it calls `to_array()`. That is also where lazy data actually gets loaded.

## Possible next steps

1. Decide open questions 1 and 2; they are small and only need a choice.
2. Try the dataset against a real zarr store to see how per-sample reads behave, then prototype batched fetching (questions 4 and 5).
3. Design duration-based windows (question 3).
