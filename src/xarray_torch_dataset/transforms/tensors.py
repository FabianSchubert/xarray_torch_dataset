"""Transforms that turn fetched xarray data into tensors."""

from collections.abc import Mapping

import torch
import xarray as xr


class ToTensors(torch.nn.Module):
    """Stack each view's variables into one float tensor (variables first).

    Takes the sample of an `XrDataset`, ``{view name: xr.Dataset}``, and
    returns ``{view name: tensor}``. The variables are stacked along a new
    first axis, in the order they have in the dataset.

    Only the values are kept, as ``float32``: coordinates, names and
    dtypes are dropped. All variables of one view must have the same
    dimensions, in the same order.
    """

    def forward(self, sample: Mapping[str, xr.Dataset]) -> dict[str, torch.Tensor]:
        out = {}
        for name, ds in sample.items():
            dims = {var: ds[var].dims for var in ds.data_vars}
            if len(set(dims.values())) > 1:
                # to_array would broadcast them to a common shape silently
                raise ValueError(
                    f"The variables of view {name!r} have different dimensions "
                    f"({dims}), so they cannot be stacked into one tensor. Put "
                    "them into separate views or use another transform."
                )
            out[name] = torch.from_numpy(ds.to_array().values).float()
        return out
