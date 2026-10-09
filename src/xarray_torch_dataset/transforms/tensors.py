"""Transforms that turn fetched xarray data into tensors."""

from collections.abc import Mapping

import torch
import xarray as xr


class ToTensors(torch.nn.Module):
    """Stack each view's variables into one float tensor (variables first).

    Takes the sample of an `XrDataset`, ``{view name: xr.Dataset}``, and
    returns ``{view name: tensor}``. All variables of one view must share
    the same dimensions. This is where lazily loaded data is read.
    """

    def forward(self, sample: Mapping[str, xr.Dataset]) -> dict[str, torch.Tensor]:
        return {
            name: torch.from_numpy(ds.to_array().values).float()
            for name, ds in sample.items()
        }
