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


# -- fetch_many -----------------------------------------------------------


def assert_matches_fetch(view, points):
    out = view.fetch_many(points)
    assert len(out) == len(points)
    for data, point in zip(out, points, strict=True):
        xr.testing.assert_identical(data, view.fetch(point))


@pytest.mark.parametrize(
    "points",
    [
        [at(time=[0, 1, 2]), at(time=[1, 2, 3]), at(time=[20, 21, 22])],
        [at(time=[5, 2, 9]), at(time=[9, 9, 2])],  # unsorted and repeated
        [at(time=4), at(time=4), at(time=30)],  # scalars drop the dimension
        [at(time=[0, 1], lat=[0, 1]), at(time=[1, 2], lat=[0, 1])],
        [at(time=[0, 1]), at(time=[3, 4, 5, 6]), at(time=7)],  # different shapes
        [at(time=[0, 1]), at(lat=[2, 3])],  # no dimension in common
        [at(time=[0, 1])],
        [],
    ],
)
def test_fetch_many_matches_fetch(source, points):
    assert_matches_fetch(Positions(source), points)


@pytest.mark.parametrize(
    "points",
    [
        [at(time=[-1, 0, 1]), at(time=[29, 30, 31]), at(time=[4, 5, 6])],
        [at(time=None), at(time=3), at(time=31)],
        [at(time=[-1, 31]), at(time=None)],  # nothing exists at all
        [at(time=[-1, 2], lat=[0, 9, 3]), at(time=[2, 3], lat=[1, 2, 3])],
    ],
)
def test_fetch_many_fills_like_fetch(source, points):
    assert_matches_fetch(Positions(source, unavailable="fill"), points)


def test_fetch_many_raises_on_missing_positions_when_rejecting(source):
    with pytest.raises(IndexError):
        Positions(source).fetch_many([at(time=[0, 1]), at(time=[30, 31])])


def test_fetch_many_reads_the_source_once(source, monkeypatch):
    view = Positions(source)
    reads = []
    isel = xr.Dataset.isel

    def counting_isel(ds, *args, **kwargs):
        if ds is view.ds:
            reads.append(args)
        return isel(ds, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "isel", counting_isel)
    view.fetch_many([at(time=[0, 1, 2]), at(time=[1, 2, 3]), at(time=[7, 8, 9])])
    assert len(reads) == 1


def test_fetch_many_does_not_read_more_than_the_points_ask_for(source, monkeypatch):
    view = Positions(source)
    # one element each; a block over both dimensions would hold nine
    points = [at(time=0, lat=0), at(time=5, lat=1), at(time=9, lat=2)]
    read = []
    isel = xr.Dataset.isel

    def recording_isel(ds, *args, **kwargs):
        out = isel(ds, *args, **kwargs)
        if ds is view.ds:
            read.append(out.a.size)
        return out

    monkeypatch.setattr(xr.Dataset, "isel", recording_isel)
    out = view.fetch_many(points)
    assert read == [1, 1, 1]
    assert [float(o.a) for o in out] == [0.0, 21.0, 38.0]


def test_fetch_many_uses_an_overridden_fetch(source):
    class Computed(View):
        def select(self, point):
            raise AssertionError("select must not be called")

        def fetch(self, point):
            return xr.Dataset({"twice": 2 * point["n"]})

    out = Computed(source).fetch_many([{"n": 1}, {"n": 4}])
    assert [int(o.twice) for o in out] == [2, 8]
