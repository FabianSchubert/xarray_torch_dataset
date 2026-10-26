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


@pytest.mark.parametrize("several", [False, True])
def test_locate_needs_unique_labels(source, times, several):
    repeated = source.isel(time=[0, 1, 1, 2])
    labels = times[[0, 1]] if several else times[1]
    with pytest.raises(ValueError, match="repeated labels along 'time'"):
        Positions(repeated).locate("time", labels)
    # other dimensions of the same source are not affected
    levels = repeated.assign_coords(lat=[10, 20, 30, 40])
    assert Positions(levels).locate("lat", 30) == 2


def test_arithmetic_on_a_failed_lookup_raises(source):
    position = Positions(source).locate("time", np.datetime64("2021-01-01"))
    with pytest.raises(TypeError):
        position + np.array([1, 2])


# -- accepts ------------------------------------------------------------


@pytest.mark.parametrize(
    ("positions", "accepted"),
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
def test_accepts_checks_positions_against_the_source(source, positions, accepted):
    assert Positions(source).accepts(at(time=positions)) is accepted


def test_accepts_needs_every_dimension_complete(source):
    view = Positions(source)
    assert view.accepts(at(time=3, lat=[0, 3]))
    assert not view.accepts(at(time=3, lat=[0, 4]))


def test_fill_view_accepts_without_selecting(source):
    class Unselectable(View):
        def select(self, point):
            raise AssertionError("select must not be called")

    assert Unselectable(source, unavailable="fill").accepts({})


def test_errors_in_select_are_not_swallowed(source):
    class Typo(View):
        def select(self, point):
            return {"time": point["tiem"]}

    with pytest.raises(KeyError):
        Typo(source).accepts({"time": 0})


# -- fetch ----------------------------------------------------------------


@pytest.mark.parametrize("unavailable", ["reject", "fill"])
def test_fetch_reads_a_complete_selection(source, unavailable):
    view = Positions(source, unavailable=unavailable)
    indexers = {"time": [5, 2, 9], "lat": 1}
    xr.testing.assert_identical(view.fetch(at(**indexers)), source.isel(indexers))


@pytest.mark.parametrize("positions", [-1, 31, None, [0, -1]])
def test_fetch_raises_on_missing_positions_when_rejecting(source, positions):
    with pytest.raises(IndexError):
        Positions(source).fetch(at(time=positions))


@pytest.mark.parametrize("unavailable", ["reject", "fill"])
def test_slices_are_rejected(source, unavailable):
    view = Positions(source, unavailable=unavailable)
    with pytest.raises(TypeError, match="slice"):
        view.fetch(at(time=slice(0, 3)))
    if unavailable == "reject":
        with pytest.raises(TypeError, match="slice"):
            view.accepts(at(time=slice(0, 3)))


# -- fetch with unavailable="fill" --------------------------------------------


def test_fill_keeps_shape_and_blanks_missing_entries(source, times):
    out = Positions(source, unavailable="fill").fetch(at(time=[-1, 0, 1, 31]))

    assert out.a.shape == (4, 4)
    assert np.isnan(out.a.values[[0, 3]]).all()
    np.testing.assert_array_equal(out.a.values[[1, 2]], source.a.values[[0, 1]])
    # an integer variable is filled too, which makes it float
    np.testing.assert_array_equal(out["count"].values, [np.nan, 0, 1, np.nan])


def test_fill_masks_coordinates_of_missing_entries(source, times):
    out = Positions(source, unavailable="fill").fetch(at(time=[-1, 0, 1, 31]))

    assert np.isnat(out.time.values[[0, 3]]).all()
    np.testing.assert_array_equal(out.time.values[[1, 2]], times[[0, 1]])
    np.testing.assert_array_equal(out.day_of_year.values, [np.nan, 1, 2, np.nan])


@pytest.mark.parametrize("position", [None, -1, 31])
def test_fill_missing_scalar_blanks_the_whole_sample(source, position):
    out = Positions(source, unavailable="fill").fetch(at(time=position))

    assert out.a.dims == ("lat",)  # the dimension is dropped as usual
    assert out.a.isnull().all()
    assert np.isnat(out.time.values)


def test_fill_combines_missing_entries_of_several_dimensions(source):
    out = Positions(source, unavailable="fill").fetch(at(time=[-1, 2], lat=[0, 9, 3]))

    expected = np.full((2, 3), np.nan)
    expected[1, [0, 2]] = source.a.values[2, [0, 3]]
    np.testing.assert_array_equal(out.a.values, expected)


def test_fill_returns_floats_whether_or_not_something_is_missing(source):
    view = Positions(source, unavailable="fill")
    complete = view.fetch(at(time=[0, 1]))
    partly = view.fetch(at(time=[0, 31]))
    nothing = view.fetch(at(time=None))

    for out in (complete, partly, nothing):
        assert out["count"].dtype == float  # an integer variable
        assert out.day_of_year.dtype == float  # an integer coordinate
        assert out.a.dtype == source.a.dtype
    np.testing.assert_array_equal(complete["count"].values, [0.0, 1.0])
    # without filling, integers stay integers
    assert Positions(source).fetch(at(time=[0, 1]))["count"].dtype.kind == "i"


@pytest.mark.parametrize("positions", [[3, 4], [3, 4, 31], [31, 32], None, 31])
def test_fill_leaves_variables_without_the_dimension_alone(source, positions):
    with_static = source.assign(static=("lat", [100.0, 200.0, 300.0, 400.0]))
    out = Positions(with_static, unavailable="fill").fetch(at(time=positions))

    assert out.static.dims == ("lat",)
    np.testing.assert_array_equal(out.static.values, with_static.static.values)
    # what does lie along time is still filled
    expected = np.isin(np.atleast_1d(positions), [None, 31, 32])
    np.testing.assert_array_equal(np.atleast_1d(out["count"].isnull()), expected)


@pytest.fixture
def recording(source):
    """`source` as dask arrays, one chunk per step, and the steps that get read."""
    import dask
    import dask.array

    read = []

    def record(block, block_info=None):
        read.append(block_info[None]["chunk-location"][0])
        return block

    def lazily(var):
        data = dask.array.map_blocks(record, var.data, dtype=var.dtype, meta=var.data)
        return var.copy(data=data)

    with dask.config.set(scheduler="synchronous"):
        yield source.chunk(time=1).map(lazily), read


def test_fill_reads_only_the_steps_that_exist(source, recording):
    lazy, read = recording
    out = Positions(lazy, ["a"], unavailable="fill").fetch(at(time=[29, 30, 31]))
    values = out.a.values

    assert sorted(set(read)) == [29, 30]  # no other step stands in
    np.testing.assert_array_equal(values[:2], source.a.values[[29, 30]])
    assert np.isnan(values[2]).all()


@pytest.mark.parametrize("unavailable", ["reject", "fill"])
def test_fetch_returns_loaded_data(recording, unavailable):
    lazy, read = recording
    out = Positions(lazy, ["a"], unavailable=unavailable).fetch(at(time=[3, 4]))

    assert sorted(set(read)) == [3, 4]  # read by fetch, not by a later access
    assert isinstance(out.a.variable.data, np.ndarray)


@pytest.mark.parametrize("positions", [None, 31, [-1, 31]])
def test_fill_reads_nothing_if_nothing_exists(recording, positions):
    lazy, read = recording
    out = Positions(lazy, unavailable="fill").fetch(at(time=positions))

    assert out.a.isnull().all() and out["count"].isnull().all()
    assert out.a.shape == np.shape(positions or 0) + (4,)
    assert read == []


def test_fill_still_reads_variables_that_are_not_missing(source, recording):
    lazy, read = recording
    static = source.a.isel(time=0, drop=True).rename("static")
    out = Positions(lazy.assign(static=static.chunk()), unavailable="fill").fetch(
        at(time=[31, 32])
    )

    np.testing.assert_array_equal(out.static.values, static.values)
    assert out.a.isnull().all() and read == []


# -- labels and available ---------------------------------------------------


def test_labels_are_those_of_the_selected_positions(source, times):
    view = Positions(source)
    assert view.labels(at(time=4)) == {"time": times[4]}
    np.testing.assert_array_equal(view.labels(at(time=[5, 2]))["time"], times[[5, 2]])


def test_labels_leave_out_dimensions_without_an_index(source, times):
    assert set(Positions(source).labels(at(time=4, lat=[0, 1]))) == {"time"}


@pytest.mark.parametrize("position", [None, 31])
def test_label_of_a_missing_position_is_missing(source, position):
    assert np.isnat(Positions(source).labels(at(time=position))["time"])


def test_labels_of_missing_entries_are_missing(source, times):
    labels = Positions(source).labels(at(time=[-1, 0, 31]))["time"]
    nat = np.datetime64("NaT", "D")
    np.testing.assert_array_equal(labels, np.array([nat, times[0], nat]))
    assert labels.dtype == source.time.dtype


@pytest.fixture
def levels():
    """A source with an integer index."""
    return xr.Dataset(
        {"t": (("level",), np.arange(3.0))}, coords={"level": [1000, 850, 500]}
    )


def test_integer_labels_become_floats_where_one_is_missing(levels):
    view = Positions(levels)
    assert view.labels(at(level=[0, 2]))["level"].dtype.kind == "i"
    np.testing.assert_array_equal(
        view.labels(at(level=[0, -1]))["level"], [1000.0, np.nan]
    )
    assert np.isnan(view.labels(at(level=None))["level"])


def test_integer_labels_are_always_floats_when_filling(levels):
    labels = Positions(levels, unavailable="fill").labels(at(level=[0, 2]))
    assert labels["level"].dtype == float


@pytest.mark.parametrize("positions", [3, None, 31, [0, 1], [-1, 0, 31], [31, 32]])
@pytest.mark.parametrize("unavailable", ["reject", "fill"])
def test_fill_puts_the_labels_on_the_coordinates(source, positions, unavailable):
    view = Positions(source, unavailable=unavailable)
    point = at(time=positions)
    labels = view.labels(point)["time"]
    if unavailable == "reject" and not view.available(point):
        return  # a rejecting view cannot fetch it
    np.testing.assert_array_equal(view.fetch(point).time.values, labels)


class Wanted(Positions):
    """`Positions` that knows the labels it asks for, e.g. from a table."""

    def labels(self, point):
        return dict(point["wanted"])


def test_fill_puts_the_labels_of_an_override_on_missing_entries(source, times):
    wanted = np.array(["2019-12-31", "2020-01-01", "2020-02-01"], dtype="datetime64[D]")
    view = Wanted(source, unavailable="fill")
    out = view.fetch({"positions": {"time": [-1, 0, 31]}, "wanted": {"time": wanted}})

    # in the dtype of the coordinate, whatever the dtype of the labels
    assert out.time.dtype == source.time.dtype
    np.testing.assert_array_equal(out.time.values, wanted)
    assert np.isnan(out.a.values[[0, 2]]).all()
    # other coordinates along the dimension are still masked
    np.testing.assert_array_equal(out.day_of_year.values, [np.nan, 1, np.nan])


def test_fill_puts_a_scalar_label_of_an_override_on_a_missing_sample(source):
    wanted = np.datetime64("2021-06-01")
    view = Wanted(source, unavailable="fill")
    out = view.fetch({"positions": {"time": None}, "wanted": {"time": wanted}})

    assert out.time.values == wanted
    assert out.a.isnull().all()


def test_complete_selections_are_not_relabelled(source, times):
    view = Wanted(source, unavailable="fill")
    point = {"positions": {"time": [0, 1]}, "wanted": {"time": times[[5, 6]]}}
    np.testing.assert_array_equal(view.fetch(point).time.values, times[[0, 1]])


@pytest.mark.parametrize(
    ("positions", "available"),
    [(0, True), ([0, 30], True), (None, False), (31, False), ([0, -1], False)],
)
@pytest.mark.parametrize("unavailable", ["reject", "fill"])
def test_available_tells_whether_something_is_missing(
    source, positions, available, unavailable
):
    view = Positions(source, unavailable=unavailable)
    assert view.available(at(time=positions)) is available
    # a filling view accepts the point all the same
    assert view.accepts(at(time=positions)) is (available or unavailable == "fill")
