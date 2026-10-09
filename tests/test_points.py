import numpy as np
import pytest

from xarray_torch_dataset import PointGrid


def test_grid_has_every_combination_with_the_last_dimension_fastest():
    grid = PointGrid(time=[10, 20], lat=[1, 2, 3])
    assert len(grid) == 6
    assert [grid[i] for i in range(6)] == [
        {"time": t, "lat": lat} for t in (10, 20) for lat in (1, 2, 3)
    ]


def test_grid_keeps_numpy_labels():
    times = np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]")
    grid = PointGrid(time=times)
    assert grid[2]["time"] == times[2]


@pytest.mark.parametrize("i", [-1, 6])
def test_grid_rejects_indices_outside_the_grid(i):
    with pytest.raises(IndexError):
        PointGrid(time=[10, 20], lat=[1, 2, 3])[i]
