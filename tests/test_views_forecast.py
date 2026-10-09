import numpy as np
import pytest
import xarray as xr

from xarray_torch_dataset import FindLatestForecastView, PointGrid, XrDataset


def hours(n):
    return np.timedelta64(n, "h")


HOURLY = np.arange("2024-10-01T00", "2024-10-03T00", dtype="datetime64[h]")  # 48
LEADS = np.arange(1, 25).astype("timedelta64[h]")
WINDOW = (hours(-2), hours(0))


def forecast(init_times, lead_times=LEADS):
    """A forecast source whose values identify (init, lead) position."""
    shape = (len(init_times), len(lead_times), 2)
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    return xr.Dataset(
        {"t2m": (("init_time", "lead_time", "x"), values)},
        coords={"init_time": init_times, "lead_time": lead_times},
    )


def expected_selection(source, point, limits):
    """Brute force: try every init time, keep the latest that fits."""
    lower, upper = limits or (None, None)
    valid_time = point["init_time"] + point["lead_time"]
    inits = source.init_time.values
    lead_positions = {lead: j for j, lead in enumerate(source.lead_time.values)}
    best = None
    for i, init in enumerate(inits):
        offset = init - point["init_time"]
        if lower is not None and offset < lower:
            continue
        if upper is not None and offset > upper:
            continue
        j = lead_positions.get(valid_time - init)
        if j is not None and (best is None or init > inits[best[0]]):
            best = (i, j)
    return best


@pytest.fixture
def three_hourly():
    return forecast(HOURLY[::3])


def point(init, lead):
    return {"init_time": HOURLY[init], "lead_time": hours(lead)}


def test_point_on_the_source_grid_selects_itself(three_hourly):
    view = FindLatestForecastView(three_hourly, init_time_offset_limits=WINDOW)
    out = view.fetch(point(init=6, lead=5))
    xr.testing.assert_identical(
        out, three_hourly.sel(init_time=HOURLY[6], lead_time=hours(5))
    )


def test_point_between_inits_uses_an_earlier_init_with_a_longer_lead(three_hourly):
    view = FindLatestForecastView(three_hourly, init_time_offset_limits=WINDOW)
    out = view.fetch(point(init=8, lead=5))  # valid at hour 13
    xr.testing.assert_identical(
        out, three_hourly.sel(init_time=HOURLY[6], lead_time=hours(7))
    )


def test_latest_init_in_the_window_wins(three_hourly):
    wide = FindLatestForecastView(
        three_hourly, init_time_offset_limits=(hours(-12), hours(0))
    )
    assert wide.select(point(init=8, lead=5))["init_time"] == 2  # hour 6, not 3 or 0


def test_window_can_exclude_the_latest_inits(three_hourly):
    older = FindLatestForecastView(
        three_hourly, init_time_offset_limits=(hours(-12), hours(-3))
    )
    assert older.select(point(init=8, lead=5)) == {"init_time": 1, "lead_time": 9}


def test_no_limits_allows_any_init_time(three_hourly):
    view = FindLatestForecastView(three_hourly)
    # valid at hour 6: the 03:00 init is the latest one that has a lead for it
    assert view.select(point(init=0, lead=6)) == {"init_time": 1, "lead_time": 2}


def test_point_without_a_matching_forecast_is_missing(three_hourly):
    view = FindLatestForecastView(three_hourly, init_time_offset_limits=WINDOW)
    unreachable = point(init=2, lead=24)  # would need a 26 h lead from hour 0

    assert view.select(unreachable) == {"init_time": None, "lead_time": None}
    assert not view.is_valid(unreachable)


def test_fill_returns_a_nan_field_for_a_missing_forecast(three_hourly):
    view = FindLatestForecastView(
        three_hourly, init_time_offset_limits=WINDOW, missing="fill"
    )
    out = view.fetch(point(init=2, lead=24))

    assert out.t2m.dims == ("x",)
    assert out.t2m.isnull().all()
    assert np.isnat(out.init_time.values) and np.isnat(out.lead_time.values)


LIMITS = [
    WINDOW,
    None,
    (None, hours(0)),
    (hours(-1), None),
    (hours(-6), hours(3)),
    (hours(-12), hours(-3)),
    (hours(1), hours(2)),
]


@pytest.mark.parametrize("limits", LIMITS, ids=str)
@pytest.mark.parametrize(
    "source",
    [
        forecast(HOURLY),
        forecast(HOURLY[::3]),
        forecast(HOURLY[::6], LEADS[2::3]),
        forecast(np.random.default_rng(0).permutation(HOURLY[::3])),
    ],
    ids=["hourly", "3-hourly", "6-hourly inits, 3-hourly leads", "unsorted inits"],
)
def test_matches_a_brute_force_search(source, limits):
    view = FindLatestForecastView(source, init_time_offset_limits=limits)
    # points start before and end after the source's init times
    inits = np.arange("2024-09-30T12", "2024-10-03T12", dtype="datetime64[h]")
    points = PointGrid(init_time=inits[::5], lead_time=LEADS[::2])

    for i in range(len(points)):
        selection = view.select(points[i])
        expected = expected_selection(source, points[i], limits)
        if expected is None:
            assert selection == {"init_time": None, "lead_time": None}
        else:
            assert (selection["init_time"], selection["lead_time"]) == expected


def test_dataset_drops_the_points_a_coarser_forecast_cannot_serve(three_hourly):
    points = PointGrid(init_time=HOURLY, lead_time=LEADS)
    views = {
        "fine": FindLatestForecastView(forecast(HOURLY), None, WINDOW),
        "coarse": FindLatestForecastView(three_hourly, None, WINDOW),
    }
    dataset = XrDataset(points, views)

    # per init time off the 3-hourly grid, the last 1 or 2 leads are unreachable
    assert len(dataset) == len(points) - 16 * (1 + 2)
