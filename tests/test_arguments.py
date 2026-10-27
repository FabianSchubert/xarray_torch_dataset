import functools
import json
from pathlib import Path

import numpy as np
import xarray as xr

from xarray_torch_dataset import OffsetView, PointGrid
from xarray_torch_dataset.arguments import describe


def test_plain_values_stay_as_they_are():
    arguments = {"none": None, "flag": True, "n": 3, "x": 0.5, "name": "fc"}
    assert describe(arguments).values == arguments


def test_numpy_scalars_and_paths_become_plain_values():
    described = describe({"n": np.int64(3), "x": np.float32(0.5), "path": Path("a")})
    assert described.values == {"n": 3, "x": 0.5, "path": "a"}
    assert type(described.values["n"]) is int


def test_lists_and_dicts_are_described_item_by_item():
    described = describe({"crop": (172, 561), "options": {"path": Path("b")}})
    assert described.values == {"crop": [172, 561], "options": {"path": "b"}}


def test_arrays_are_kept_in_full_and_referred_to_by_their_path():
    times = np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]")
    offset = np.timedelta64(2, "h")
    described = describe({"times": times, "nested": {"offset": offset}})

    assert described.values == {
        "times": {"array": "times"},
        "nested": {"offset": {"array": "nested.offset"}},
    }
    np.testing.assert_array_equal(described.arrays["times"], times)
    assert described.arrays["nested.offset"] == offset


def test_functions_and_classes_are_described_by_name():
    described = describe({"function": np.mean, "class": PointGrid})
    assert described.values == {
        "function": {"function": "numpy.mean"},
        "class": {"function": "xarray_torch_dataset.points.PointGrid"},
    }


def test_a_partial_is_described_by_its_function_and_arguments():
    truncate = functools.partial(np.round, decimals=2)
    assert describe({"reduce": truncate}).values == {
        "reduce": {
            "function": "numpy.round",
            "args": [],
            "keywords": {"decimals": 2},
        }
    }


class Filter:
    """An object with an array among its attributes, like a filter of times."""

    def __init__(self, valid_times, every):
        self.valid_times = valid_times
        self.every = every


def test_objects_are_described_by_class_and_attributes():
    times = np.arange("2024-01-01", "2024-01-03", dtype="datetime64[h]")
    described = describe({"filter": Filter(times, every=6)})

    assert described.values == {
        "filter": {
            "class": f"{__name__}.Filter",
            "attributes": {"valid_times": {"array": "filter.valid_times"}, "every": 6},
        }
    }
    np.testing.assert_array_equal(described.arrays["filter.valid_times"], times)


def test_xarray_objects_are_left_out_and_listed(source, times):
    view = OffsetView(source, ["a"], unavailable="fill")
    described = describe({"points": PointGrid(time=times), "views": {"x": view}})

    assert described.left_out == ["views.x.ds"]
    # what is around them is described as usual
    attributes = described.values["views"]["x"]["attributes"]
    assert attributes["unavailable"] == "fill"
    np.testing.assert_array_equal(described.arrays["points.coords.time"], times)


def test_an_object_that_refers_to_itself_is_described_once():
    looped = Filter(None, every=1)
    looped.valid_times = looped
    described = describe({"filter": looped})
    assert described.left_out == ["filter.valid_times"]


def test_the_values_can_be_stored_as_json_and_are_the_same_each_time():
    arguments = {
        "times": np.arange(3),
        "filter": Filter(np.arange(2), every=2),
        "reduce": functools.partial(np.round, decimals=1),
        "path": Path("c"),
        "data": xr.Dataset(),
    }
    first, second = describe(arguments), describe(arguments)
    assert json.loads(json.dumps(first.values)) == first.values
    assert first.values == second.values
    assert first.arrays.keys() == second.arrays.keys()
