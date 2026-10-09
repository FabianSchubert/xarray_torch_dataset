"""Ready-made views."""

from xarray_torch_dataset.views.forecast import FindLatestForecastView
from xarray_torch_dataset.views.offset import OffsetView
from xarray_torch_dataset.views.weekday import SameWeekdayView

__all__ = ["FindLatestForecastView", "OffsetView", "SameWeekdayView"]
