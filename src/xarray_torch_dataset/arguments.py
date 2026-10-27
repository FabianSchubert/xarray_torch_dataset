"""Describing the arguments of a dataset, so that a cache can store and check them.

`describe` turns the arguments a dataset was constructed with (see
`XrDataset.arguments`) into a description that can be stored and compared:

- plain values (``None``, bools, numbers, strings), and lists, tuples and
  dicts of them, stay as they are; a `Path` becomes a string;
- numpy arrays and datetime or timedelta values are kept in full, as arrays,
  and referred to by their path, such as ``"filter.valid_times"``;
- functions and classes are described by module and qualified name, and a
  ``functools.partial`` by its function and its arguments;
- other objects by their class and, in the same way, their attributes;
- xarray objects, such as the source of a view, are left out, as they cannot
  be reduced to a description cheaply; so is anything else that has no
  attributes to describe it by.
"""

import functools
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


@dataclass
class Description:
    """What `describe` makes of arguments.

    `values` holds the description as plain values that can be stored as
    JSON, with ``{"array": path}`` in the place of an array; `arrays` holds
    those arrays by their path; `left_out` lists the paths of what could not
    be described.
    """

    values: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    left_out: list[str] = field(default_factory=list)


def describe(arguments: Mapping[str, Any]) -> Description:
    """Describe the arguments of a dataset, see the module."""
    description = Description(values={})
    description.values = {
        name: _describe(value, name, description, set())
        for name, value in arguments.items()
    }
    return description


def _describe(value: Any, path: str, description: Description, seen: set[int]) -> Any:
    """The description of `value` at `path`, adding arrays and what is left out."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    # before the numbers, as numpy counts timedelta64 among the integers
    if isinstance(value, np.ndarray | np.datetime64 | np.timedelta64):
        description.arrays[path] = np.asarray(value)
        return {"array": path}
    if isinstance(value, np.bool_ | np.integer | np.floating):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, xr.Dataset | xr.DataArray | xr.Variable):
        description.left_out.append(path)
        return {"left out": type(value).__name__}
    if isinstance(value, functools.partial):
        return {
            "function": _name(value.func),
            "args": _describe(list(value.args), f"{path}.args", description, seen),
            "keywords": _describe(
                value.keywords, f"{path}.keywords", description, seen
            ),
        }
    if inspect.isroutine(value) or inspect.isclass(value):
        return {"function": _name(value)}
    if isinstance(value, list | tuple):
        return [
            _describe(item, f"{path}.{i}", description, seen)
            for i, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _describe(item, f"{path}.{key}", description, seen)
            for key, item in value.items()
        }
    # an object by its class and attributes, unless it refers back to itself
    if not hasattr(value, "__dict__") or id(value) in seen:
        description.left_out.append(path)
        return {"left out": type(value).__name__}
    return {
        "class": _name(type(value)),
        "attributes": _describe(vars(value), path, description, seen | {id(value)}),
    }


def differences(recorded: Description, given: Description) -> list[str]:
    """The paths at which two descriptions of arguments differ.

    Plain values must be equal, and arrays have the same shape and values,
    whatever their unit of time. A path that only one of them has differs,
    as does a list of another length. What was left out of either is not
    compared.
    """
    found: list[str] = []
    for name in recorded.values.keys() | given.values.keys():
        _compare(
            recorded.values.get(name, _ABSENT),
            given.values.get(name, _ABSENT),
            name,
            recorded,
            given,
            found,
        )
    return sorted(found)


# stands for a path that one of two descriptions does not have
_ABSENT = object()


def _compare(
    a: Any, b: Any, path: str, recorded: Description, given: Description, found: list
) -> None:
    """Add `path` to `found` where the described values `a` and `b` differ."""
    if _is(a, "left out") or _is(b, "left out"):
        return
    if _is(a, "array") and _is(b, "array"):
        x, y = recorded.arrays[a["array"]], given.arrays[b["array"]]
        if x.shape != y.shape or not np.array_equal(x, y):
            found.append(path)
    elif _is_object(a) and _is_object(b):
        # an object is named by its own path, its attributes below it, as
        # `describe` names their arrays
        if a["class"] != b["class"]:
            found.append(path)
        else:
            _compare(a["attributes"], b["attributes"], path, recorded, given, found)
    elif isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in a.keys() | b.keys():
            _compare(
                a.get(key, _ABSENT),
                b.get(key, _ABSENT),
                f"{path}.{key}",
                recorded,
                given,
                found,
            )
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _compare(x, y, f"{path}.{i}", recorded, given, found)
    elif a is _ABSENT or b is _ABSENT or a != b:
        found.append(path)


def _is_object(value: Any) -> bool:
    """Whether a described value stands for an object by class and attributes."""
    return isinstance(value, Mapping) and set(value) == {"class", "attributes"}


def _is(value: Any, kind: str) -> bool:
    """Whether a described value stands for an array or something left out."""
    return isinstance(value, Mapping) and list(value) == [kind]


def _name(function: Any) -> str:
    """The module and qualified name of a function or class."""
    return f"{function.__module__}.{function.__qualname__}"
