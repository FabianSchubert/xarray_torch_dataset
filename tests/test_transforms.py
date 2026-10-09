import numpy as np
import torch

from xarray_torch_dataset import ToTensors


def test_to_tensors_stacks_variables_first(source):
    sample = {"x": source[["a"]].isel(time=[0, 1, 2]), "y": source.isel(time=5, lat=0)}
    out = ToTensors()(sample)

    assert set(out) == {"x", "y"}
    assert out["x"].shape == (1, 3, 4)  # variable, time, lat
    assert out["x"].dtype == torch.float32
    np.testing.assert_array_equal(out["x"][0], source.a.values[:3])
    # two scalar variables, in the order of the dataset
    assert out["y"].tolist() == [20.0, 5.0]


def test_to_tensors_keeps_nan(source):
    filled = source[["a"]].isel(time=[0, 1]).where(False)
    assert torch.isnan(ToTensors()({"x": filled})["x"]).all()
