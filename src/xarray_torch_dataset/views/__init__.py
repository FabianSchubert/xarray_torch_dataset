"""Ready-made views."""

from xarray_torch_dataset.views.forecast import FindLatestForecastView
from xarray_torch_dataset.views.offset import OffsetView

__all__ = ["FindLatestForecastView", "OffsetView"]
