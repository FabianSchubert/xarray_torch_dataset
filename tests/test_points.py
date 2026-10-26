import numpy as np
import pytest

from xarray_torch_dataset import PointGrid, PointTable


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


def test_table_has_its_rows_as_points():
    table = PointTable(time=[10, 10, 20], lat=[3, 1, 3])
    assert len(table) == 3
    assert [table[i] for i in range(3)] == [
        {"time": 10, "lat": 3},
        {"time": 10, "lat": 1},
        {"time": 20, "lat": 3},
    ]


def test_table_holds_the_points_of_a_grid_that_pass_a_filter():
    grid = PointGrid(time=[10, 20], lat=[1, 2, 3])
    kept = [i for i in range(len(grid)) if grid[i]["lat"] != 2]
    table = PointTable(**{dim: [grid[i][dim] for i in kept] for dim in ("time", "lat")})
    assert [table[i] for i in range(len(table))] == [grid[i] for i in kept]


def test_table_needs_label_arrays_of_equal_length():
    with pytest.raises(ValueError, match="differ in length"):
        PointTable(time=[10, 20], lat=[1, 2, 3])


@pytest.mark.parametrize("i", [-1, 3])
def test_table_rejects_indices_outside_the_table(i):
    with pytest.raises(IndexError):
        PointTable(time=[10, 20, 30])[i]
