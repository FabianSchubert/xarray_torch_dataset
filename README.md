# xarray_torch_dataset

PyTorch datasets on top of xarray, built from small, swappable parts.

This is an experiment. The interfaces are still changing; `STATUS.md` records the current state, the design decisions and the open questions.

## The idea

A sample is described in three steps:

```mermaid
flowchart LR
    loader["DataLoader"] -- "index i" --> point
    points["Points<br/>PointGrid or list of dicts"] -- "points[indices[i]]" --> point

    subgraph dataset["XrDataset[i]"]
        point["point<br/>{dim: label}"]
        subgraph view["view.fetch(point), one view per source"]
            select["select(point)<br/>no reading"]
            read["read at the positions,<br/>or fill with NaN"]
        end
        transform["transform<br/>e.g. ToTensors"]
    end

    source[("xarray source<br/>e.g. a zarr store")]
    point --> select
    select -- "positions" --> read
    source --> read
    read -- "{view name: xr.Dataset}" --> transform
    transform -- "sample" --> loader
```

- A **point** says where a sample is, as coordinate labels: `{"time": np.datetime64("2020-01-05")}`.
- A **view** says what to read from one xarray source for a point, for example the four time steps up to the point. Each source gets its own view, so sources with different grids or resolutions can be combined.
- A **transform**, a method of the dataset, turns the fetched xarray data into what the model needs, usually tensors.

Views only compute positions; they do not read. The dataset can therefore check at construction which points every view can serve, and it keeps only those.

```mermaid
flowchart LR
    points["Points"] --> accepts{"every view<br/>accepts(point)?"}
    accepts -- "yes" --> indices["indices: the samples"]
    accepts -- "no" --> rejected["dropped, or an error<br/>with rejected=&quot;raise&quot;"]
    indices --> collect["labels(name), available(name)<br/>per sample, without reading"]
```

## Example

```python
import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset import OffsetView, PointGrid, ToTensors, XrDataset

times = np.arange("2020-01-01", "2020-02-01", dtype="datetime64[D]")
dims = ("time", "lat", "lon")
inputs = xr.Dataset(
    {"t2m": (dims, np.random.rand(31, 8, 8)), "u10": (dims, np.random.rand(31, 8, 8))},
    coords={"time": times},
)
targets = xr.Dataset(
    {"precip": (dims, np.random.rand(31, 8, 8))}, coords={"time": times}
)

last_four_days = np.arange(-3, 1) * np.timedelta64(1, "D")


class Weather(XrDataset):
    def transform(self, sample):
        return ToTensors()(sample)


dataset = Weather(
    points=PointGrid(time=times),
    views={
        # the point's day and the three days before it
        "x": OffsetView(inputs, ["t2m", "u10"], offsets={"time": last_four_days}),
        # the point's day only
        "y": OffsetView(targets, ["precip"]),
    },
)

print(len(dataset))  # 28: the first three days have no full history
batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))
print({name: tensor.shape for name, tensor in batch.items()})
# {'x': torch.Size([4, 2, 4, 8, 8]), 'y': torch.Size([4, 1, 8, 8])}
```

`examples/` has two longer scripts: `history_windows.py` and `forecasts.py`.

## The parts

**Points.** Anything with `__len__` and `__getitem__(i)` that returns a point will do, including a plain list of dicts. `PointGrid(time=times, lat=lats)` gives every combination of the labels without building the list. `PointTable(time=times, lat=lats)` gives the rows of label arrays of equal length, for any other set of points, such as the points of a grid that pass a filter.

**Views.** A view is usually written for the application at hand (see below). Two are included, as examples and for the common cases:

| View | Selects |
|---|---|
| `OffsetView` | Fixed offsets from the point's labels, such as the last four hours. Offsets are added to the label and looked up in the source, so along time they are durations, not numbers of steps. Dimensions without offsets are selected at exactly the point's label. |
| `FindLatestForecastView` | From a forecast source with `init_time` and `lead_time`, the most recent forecast for the point's valid time. This aligns, say, a 3-hourly forecast with hourly points. |

**What a view does for a point.** The dataset calls `view.fetch(point)`, which runs as below. One question decides everything: does every position that `select` returned lie inside the source? If so, the view reads there. If not, its `unavailable` setting decides: `"reject"` raises, which the dataset avoids by dropping such points at construction, and `"fill"` reads what exists and fills the rest. A view's author writes `select`, and `labels` only to report what the view asked for where the source lacks it; the base class does the rest.

```mermaid
flowchart TD
    fetch(["view.fetch(point)"]) --> select["positions = select(point)<br/>the view's own logic, reads nothing"]
    select -- "positions" --> complete{"every position inside the source?<br/>None and -1 are not<br/>(what available(point) asks)"}
    complete -- "yes" --> read["read at the positions"]
    complete -- "no" --> setting{"the view's setting<br/>unavailable="}
    setting -- "&quot;reject&quot;" --> error["IndexError<br/>a validated dataset has dropped<br/>such points already"]
    setting -- "&quot;fill&quot;" --> fill["read only the positions that exist;<br/>missing entries get NaN as data<br/>and labels(point) as coordinates"]
    read --> loaded(["loaded xr.Dataset"])
    fill --> loaded
```

**Unavailable data.** A source may not have everything a point asks for: a history window can reach back before the source starts, or the source may lack the point's label. What happens then is decided in two places.

- Each view takes `unavailable="reject"` (the default) or `unavailable="fill"`. With `"reject"`, the view rejects such a point. With `"fill"`, it accepts every point and returns NaN for whatever the source lacks, in the shape a complete selection would have. `view.accepts(point)` gives this answer; the dataset asks it of every view for every point at construction, to decide which points to keep. Override it only for a cheaper check.
- The dataset takes `rejected="drop"` (the default) or `rejected="raise"`. With `"drop"`, points that some view rejected are left out. With `"raise"`, they are an error at construction, which helps to catch a misaligned source.

`dataset.n_rejected` says how many points each view rejected, for example `{"x": 3, "y": 0}`. A few points rejected by a history view are an edge effect; most points rejected by one view usually mean that its source is on a different grid. A point rejected by several views is counted for the first of them.

Asking every view about every point happens when the dataset is constructed and takes time on large point sets (roughly 0.1 ms per point and view). `XrDataset(..., validate=False)` skips it. Every point is then kept, and one that a view would have rejected raises an `IndexError` when it is fetched, which may be well into an epoch. It never returns wrong data. Use it when you already know that every point can be served.

A filling view never rejects a point, so the other views decide which points the dataset has. If every view fills, no point is rejected at all, and a point that no source has data for becomes a sample of NaN; keep at least one rejecting view, typically the target, to avoid that. Because integers cannot hold NaN, a filling view returns integer and boolean values as floats, for every point.

Whether a view has all data for a point, and which labels it selects, can be asked without reading anything. `view.available(point)` says whether the view has all data for the point. For a rejecting view this is the same answer as `accepts`; a filling view accepts every point, so only `available` tells whether something would be filled. `view.labels(point)` gives the coordinate labels of the selection per dimension. For the whole dataset, `dataset.available("x")` and `dataset.labels("x")` give them for every sample, as arrays with one row per sample.

All of this is only about data the source does not have: NaN values stored in a source are returned as they are, and a point that lacks a label a view needs is an error.

**Transforms.** A subclass of `XrDataset` overrides the method `transform(sample)`, which receives `{view name: xr.Dataset}` and returns the sample. The data is already in memory at that point: reading is the views' job, and a transform only reshapes what was read. Because it gets xarray objects, it can use names, dimensions and coordinates to build what the model needs. A second method, `prepare`, can do the expensive part of that work before `transform`, so that it can be cached (see "Caching samples").

`ToTensors` is the one transform included, to be called from `transform`. It stacks the variables of each view into one `float32` tensor, variables first, and keeps only the values. The variables of a view must have the same dimensions; it raises otherwise instead of broadcasting them.

A `DataLoader` needs a transform that returns tensors or arrays, because torch cannot batch xarray objects. A plain `XrDataset`, which overrides nothing, returns the xarray data itself, which is useful for looking at samples.

## Writing a view

Subclass `View` and implement `select`, which returns positions per dimension as `xr.Dataset.isel` accepts them. `locate` turns coordinate labels into positions.

```python
import numpy as np

from xarray_torch_dataset import View


class DayBeforeView(View):
    """The source's value one day before the point."""

    def select(self, point):
        return {"time": self.locate("time", point["time"] - np.timedelta64(1, "D"))}
```

A label that the source lacks gives the position `None` (or `-1` inside an array of positions), which marks the data as missing. The labels along a dimension that a view looks up must be unique; `locate` raises otherwise. `select` should mark missing data this way rather than raise, and it should not read any data.

`locate` uses the index that xarray keeps for every dimension coordinate, a pandas `Index` (`ds.indexes[dim]`): `get_loc` for a single label and `get_indexer` for several. It therefore matches labels exactly, without a tolerance, and works only along dimensions that have a coordinate. A view that needs another kind of lookup, such as the nearest label or a search in sorted labels, can compute the positions itself, as `FindLatestForecastView` does with `np.searchsorted`.

A filled entry has NaN as its data and, by default, a missing label (NaT or NaN) as its coordinate. A view that knows which label it asked for although the source lacks it can say so by overriding `labels`; filling then puts that label on the coordinate. `OffsetView` does this, so a history window that reaches back before the source starts still shows the days it covers. A view that matches forecasts to points could report the run it chose and the lead time it needed in the same way, also when that run does not reach far enough.

Where a position exists, an override must report the source's label there, so that labels and positions describe the same selection. `fetch` checks this and raises a `ValueError` if they disagree. The coordinates of what `fetch` returns are therefore always `labels(point)`, whether the data was read or filled. The simplest way to get this right is to compute the labels and let `select` locate them, as in the example below.

```python
class DayBeforeView(View):
    """The source's value one day before the point, labelled by that day."""

    def labels(self, point):
        return {"time": point["time"] - np.timedelta64(1, "D")}

    def select(self, point):
        return {"time": self.locate("time", self.labels(point)["time"])}
```

## Caching samples

Reading and reducing the sources can take much longer than training on the result. A cache stores the expensive part once. For that, a subclass of `XrDataset` overrides two methods:

- `prepare(data)` does the expensive part and returns one `xr.Dataset` per sample, the **prepared sample**: for example the mean and spread over the members of an ensemble.
- `transform(sample)` turns a prepared sample into what the model gets: for example tensors.

`dataset.cache(path)` writes the prepared sample of every point. `Subclass.from_cache(path)` reads them back, one row per point, into an instance of the same class, so that its `transform` gives the same samples as the dataset on the sources:

```mermaid
flowchart LR
    subgraph raw["Subclass(...): from the sources"]
        views["views<br/>{name: xr.Dataset}"] --> prepare["prepare<br/>the expensive part"]
    end
    prepare -- "prepared sample: xr.Dataset" --> transform["transform<br/>the cheap part"]
    prepare -. "dataset.cache(path)" .-> store[("cache store<br/>one row per point")]
    store -- "Subclass.from_cache(path)" --> transform
    transform --> sample["sample"]
```

```python
import numpy as np
import torch
import xarray as xr

from xarray_torch_dataset import OffsetView, PointGrid, XrDataset

times = np.arange("2020-01-01", "2020-02-01", dtype="datetime64[D]")
members = xr.Dataset(
    {"t2m": (("time", "member", "lat", "lon"), np.random.rand(31, 10, 8, 8))},
    coords={"time": times},
)


class Ensemble(XrDataset):
    def __init__(self, members):
        super().__init__(PointGrid(time=times), {"x": OffsetView(members, ["t2m"])})

    def prepare(self, data):
        """The expensive part: the mean and spread over the members."""
        t2m = data["x"].t2m
        return xr.Dataset(
            {"mean": t2m.mean("member"), "std": t2m.std("member")}
        ).drop_vars("time")

    def transform(self, sample):
        """The cheap part: the tensor the model gets."""
        return torch.from_numpy(
            np.stack([sample["mean"].values, sample["std"].values])
        ).float()


raw = Ensemble(members)
raw.cache("t2m.zarr")

cached = Ensemble.from_cache("t2m.zarr")
assert all(torch.equal(raw[i], cached[i]) for i in range(len(raw)))
```

What a prepared sample may hold:

- **Data variables** are what differs between samples. Each becomes an array of the store with a leading `sample` dimension. They must have the same names, dimensions, shapes and dtypes in every sample.
- **Coordinates** are stored once and must be the same in every sample, so `cache` raises otherwise. Labels that differ between samples, such as the time a view fetched, have to be dropped or become data variables; the example drops `time`. Names of the dimensions of the points cannot be used, as the store holds the points under them.

The store is a zarr 3 store that xarray and plain zarr can read. Large variables get one chunk per sample and several samples per file (a shard, about 1 GB by default), small ones such as times are one chunk. The labels of the points are stored too, one array per dimension of the points, and the `attrs` of `cache(path, attrs=...)` become the attributes of the store, for example the settings it was written with, to check before using it (`CacheView(path).attrs`).

`from_cache` reads every point of the cache, in their order. `from_cache(path, points=...)` reads those points instead; a point the cache does not hold is rejected, or raises with `rejected="raise"`. The instance it returns is not built by `__init__`, which opens the sources a cached dataset does not need. So `transform` must not use what `__init__` sets; it may use `self.arguments`, the arguments the dataset was constructed with, which a dataset read with its arguments (below) has as well. `CacheView(path)` is the view underneath, for reading the rows of a cache directly; its rows also have the point's labels as coordinates.

**Checking a cache against its arguments.** Every dataset keeps the arguments it was constructed with as `self.arguments`, by name and with the defaults of its constructor, and `cache` records them with the store: plain values as they are, arrays in full, functions by module and name, other objects by their class and attributes. `Ensemble.from_cache(path, arguments={"members": members})` compares the arguments a dataset would be constructed with to that record, and raises a `ValueError` that names each one that differs, for example `crop_y` or `filter.valid_times`. That catches a cache which no longer belongs to the arguments, say after a configuration changed. xarray objects, like the `members` of the example, cannot be recorded: they are named in a warning and not checked, so a subclass that takes the paths of its stores is checked more fully than one that takes the data. Without `arguments`, nothing is checked, and `self.arguments` holds the record. `Ensemble.open(cache, **arguments)` combines both ways: it reads the cache with `from_cache` and checks it if the store exists, and otherwise constructs the dataset from the arguments, with a warning that the cache does not exist yet. The record marks arrays, objects and what was left out with small dicts, `{"array": ...}`, `{"left out": ...}`, and `{"class": ..., "attributes": ...}`; a dict argument of exactly one of these shapes would be misread.

`dataset.cache(path, num_workers=4, multiprocessing_context="forkserver")` reads with DataLoader workers. Start them with `forkserver` rather than by forking, as zarr runs threads that a forked worker may deadlock on; the dataset's class and views then have to be importable from a module.

## Reading from zarr

Open stores without dask:

```python
ds = xr.open_dataset("data.zarr", engine="zarr", chunks=None)
```

`xr.open_zarr` and `chunks={}` give dask-backed data, which has two drawbacks here:

- **It can hang.** If the main process reads from a dask-backed dataset (looking at `dataset[0]` is enough) and a `DataLoader` then starts workers by forking, which is the default on Linux, the first batch never arrives.
- **It was not faster.** In the benchmarks, dask was slower on small and medium chunks and about equal on large ones.

If you need dask, start the workers with `DataLoader(..., multiprocessing_context="forkserver")` or use dask's synchronous scheduler (`dask.config.set(scheduler="synchronous")`).

The benchmark results are in `STATUS.md`, and the scripts in `benchmarks/`.

## Development

```bash
git clone git@gitlab.dkrz.de:b383481/xarray_torch_dataset.git
cd xarray_torch_dataset
uv sync
uv run pre-commit install
uv run pytest
```

Pre-commit runs `ruff`, `mypy` and `pytest`. `STATUS.md` has more on the setup.
