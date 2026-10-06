# Project status

Snapshot as of 2026-10-06. This file records where the project stands and which design questions are open, so work can continue on another machine.

## What this project is

An experiment in building PyTorch datasets on top of xarray from small, swappable parts:

```
index --Sampler--> point (in the Anchor's sampling space)
point --View-----> data from one xarray source   (one View per source)
{name: data} --Transform--> returned sample
```

A "point" is whatever the user's anchor, sampler and views agree on (for example a dict of coordinates such as `{"time": ...}`). The framework never looks inside it.

## Current state

- **Base classes** live in `src/xarray_torch_dataset/base.py` and are re-exported from the package root:
  - `Anchor`: a finite, indexable set of points (the sampling space).
  - `Sampler`: maps a dataset index to a point; wraps an anchor.
  - `View`: pulls data for a point from one `xr.Dataset`.
  - `XrDataset`: a `torch.utils.data.Dataset` that asks the sampler for a point, calls every view, and applies an optional transform to the resulting `{name: data}` dict.
- **`example.py`** contains example implementations (`TimeAnchor`, `StrideSampler`, `HistoryView`, `AtTimeView`, `ToTensors`) and a usage block on random in-memory data. It runs and prints:
  ```
  14 {'x': torch.Size([4, 2, 4, 8, 8]), 'y': torch.Size([4, 1, 8, 8])}
  ```
  That is 14 samples; a batch of 4 gives `x` as (batch, 2 variables, 4 history steps, 8, 8) and `y` as (batch, 1 variable, 8, 8).
- **Dependencies**: `numpy`, `torch`, `xarray`. Dev dependencies: `ruff`, `mypy`, `pre-commit`.
- **Tooling**: pre-commit runs `ruff check --fix`, `ruff format` and `mypy` as local hooks through `uv run`, so tool versions come from `uv.lock`. Ruff lint rules are `E`, `F`, `I`, `UP`, `B`. mypy checks `src` only, with `check_untyped_defs` and `warn_unused_ignores` but not `strict`.
- **No tests**, by decision (see `CLAUDE.md`).
- So far the code has only been exercised on small in-memory data, never on zarr or dask-backed sources.

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

## Open design questions

These are critique points on the current design. None has been acted on yet; the base classes were moved into the package unchanged.

### Redundancies

1. **`Sampler` and `Anchor` have the same interface.** Both are `__len__` plus `__getitem__` returning a point, so a sampler is effectively an anchor wrapping another anchor. `XrDataset` only talks to the sampler and never touches `sampler.anchor`, so the anchor is an implementation detail of one sampler rather than a separate concept in the framework.
2. **`Sampler` overlaps with what torch already provides.** Choosing which indices to visit (stride, offset, subset, shuffle) is the job of `torch.utils.data.Sampler` and `Subset`. The name also collides with torch's `Sampler`, which has a different contract (it yields indices, not points).
3. **`View.__init__` stores `ds`, but the contract does not need it.** The framework only calls `view(point)`. Holding one `xr.Dataset` is a convenience for the example views.

### Ambiguities

4. **Who decides which points are valid.** In the example, `StrideSampler(offset=history - 1)` has to be matched by hand to `HistoryView(n=history)`. If they disagree, the slice start in `HistoryView` goes negative and `isel` silently returns an empty or wrong window instead of raising.
5. **What a "point" is.** Nothing checks the agreement between anchor and views. A view expecting `point["time"]` paired with an anchor that yields something else fails only at fetch time, with a `KeyError` from inside the view.
6. **What a view returns.** The example views return `xr.Dataset`, but the base class does not say so, and the transform has to know. `ToTensors` further assumes all variables in a view share the same dims, since it calls `to_array()`.
7. **How sources are aligned.** The anchor is built from `inputs.time`, while `targets` is a separate source assumed to share those times. `AtTimeView` would raise on a missing time, but `HistoryView` counts positional steps rather than a duration, so a gap in the time axis silently yields a window spanning the wrong period.

### Other points

8. **`StrideSampler.__len__` can undercount** (a bug in `example.py`, not a design question). `(len - offset) // stride` drops the last valid point when the division is not exact: 5 remaining points with stride 2 gives 2, but indices 0, 1 and 2 are valid. It happens to be right in the demo (28 / 2 = 14).
9. **I/O happens per sample.** Each `__getitem__` does an index lookup plus one small read per view, and `ToTensors` is where lazy data actually gets loaded (`.values`). That is fine in memory; on zarr or dask-backed data with DataLoader workers it will likely be the bottleneck, and batch-level fetching may be worth designing in early.

### Suggested starting point

Points 1 and 4 are linked. If the anchor were derived from the views' requirements (for example, each view reports the margin it needs), the anchor would own validity, and `Sampler` could either go away or shrink to pure index selection, which torch already covers. Settling that first is recommended, before more views are built on the current contracts.

## Possible next steps

1. Decide the validity question and the `Sampler` / `Anchor` overlap (points 1, 2, 4).
2. Pin down the contracts for points and view return types (points 5, 6).
3. Fix `StrideSampler.__len__`, if `StrideSampler` survives step 1.
4. Try the dataset against a real zarr store to see how per-sample reads behave (point 9).
