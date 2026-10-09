import numpy as np
import pytest
import xarray as xr
from conftest import Positions, at

from xarray_torch_dataset import View


def test_variables_restrict_the_source(source):
    assert set(Positions(source, ["a"]).ds.data_vars) == {"a"}
    assert set(Positions(source).ds.data_vars) == {"a", "count"}


# -- locate ---------------------------------------------------------------


def test_locate_scalar_label(source, times):
    assert Positions(source).locate("time", times[4]) == 4


def test_locate_missing_scalar_label_is_none(source):
    assert Positions(source).locate("time", np.datetime64("2021-01-01")) is None


def test_locate_several_labels_marks_missing_with_minus_one(source, times):
    labels = [times[2], np.datetime64("2021-01-01"), times[0]]
    assert Positions(source).locate("time", labels).tolist() == [2, -1, 0]


def test_arithmetic_on_a_failed_lookup_raises(source):
    position = Positions(source).locate("time", np.datetime64("2021-01-01"))
    with pytest.raises(TypeError):
        position + np.array([1, 2])


# -- is_valid -------------------------------------------------------------


@pytest.mark.parametrize(
    ("positions", "valid"),
    [
        (0, True),
        (30, True),
        (31, False),  # past the end
        (-1, False),  # negative positions do not count from the end
        (None, False),
        ([0, 30], True),
        ([0, -1], False),
        ([0, 31], False),
        ([], True),
    ],
)
def test_is_valid_checks_positions_against_the_source(source, positions, valid):
    assert Positions(source).is_valid(at(time=positions)) is valid


def test_is_valid_needs_every_dimension_complete(source):
    view = Positions(source)
    assert view.is_valid(at(time=3, lat=[0, 3]))
    assert not view.is_valid(at(time=3, lat=[0, 4]))


def test_fill_view_is_valid_without_selecting(source):
    class Unselectable(View):
        def select(self, point):
            raise AssertionError("select must not be called")

    assert Unselectable(source, missing="fill").is_valid({})


def test_errors_in_select_are_not_swallowed(source):
    class Typo(View):
        def select(self, point):
            return {"time": point["tiem"]}

    with pytest.raises(KeyError):
        Typo(source).is_valid({"time": 0})


# -- fetch ----------------------------------------------------------------


@pytest.mark.parametrize("missing", ["drop", "fill"])
def test_fetch_reads_a_complete_selection(source, missing):
    view = Positions(source, missing=missing)
    indexers = {"time": [5, 2, 9], "lat": 1}
    xr.testing.assert_identical(view.fetch(at(**indexers)), source.isel(indexers))


@pytest.mark.parametrize("positions", [-1, 31, None, [0, -1]])
def test_fetch_raises_on_missing_positions_when_dropping(source, positions):
    with pytest.raises(IndexError):
        Positions(source).fetch(at(time=positions))


@pytest.mark.parametrize("missing", ["drop", "fill"])
def test_slices_are_rejected(source, missing):
    view = Positions(source, missing=missing)
    with pytest.raises(TypeError, match="slice"):
        view.fetch(at(time=slice(0, 3)))
    if missing == "drop":
        with pytest.raises(TypeError, match="slice"):
            view.is_valid(at(time=slice(0, 3)))


# -- fetch with missing="fill" --------------------------------------------


def test_fill_keeps_shape_and_blanks_missing_entries(source, times):
    out = Positions(source, missing="fill").fetch(at(time=[-1, 0, 1, 31]))

    assert out.a.shape == (4, 4)
    assert np.isnan(out.a.values[[0, 3]]).all()
    np.testing.assert_array_equal(out.a.values[[1, 2]], source.a.values[[0, 1]])
    # an integer variable is filled too, which makes it float
    np.testing.assert_array_equal(out["count"].values, [np.nan, 0, 1, np.nan])


def test_fill_masks_coordinates_of_missing_entries(source, times):
    out = Positions(source, missing="fill").fetch(at(time=[-1, 0, 1, 31]))

    assert np.isnat(out.time.values[[0, 3]]).all()
    np.testing.assert_array_equal(out.time.values[[1, 2]], times[[0, 1]])
    np.testing.assert_array_equal(out.day_of_year.values, [np.nan, 1, 2, np.nan])


@pytest.mark.parametrize("position", [None, -1, 31])
def test_fill_missing_scalar_blanks_the_whole_sample(source, position):
    out = Positions(source, missing="fill").fetch(at(time=position))

    assert out.a.dims == ("lat",)  # the dimension is dropped as usual
    assert out.a.isnull().all()
    assert np.isnat(out.time.values)


def test_fill_combines_missing_entries_of_several_dimensions(source):
    out = Positions(source, missing="fill").fetch(at(time=[-1, 2], lat=[0, 9, 3]))

    expected = np.full((2, 3), np.nan)
    expected[1, [0, 2]] = source.a.values[2, [0, 3]]
    np.testing.assert_array_equal(out.a.values, expected)
