r"""
Prophet forecaster (Meta / Facebook Prophet).
=============================================

Prophet models a series as an additive decomposition

    .. math:: y(t) = g(t) + s(t) + h(t) + \varepsilon_t

with a piecewise-linear trend :math:`g(t)`, Fourier seasonalities :math:`s(t)`
(weekly + yearly here), a holiday component :math:`h(t)`, and noise. It is
robust to missing data and gives calibrated uncertainty intervals out of the
box, which makes it a strong, low-tuning benchmark for ATM demand.

We register the Indian salary-day windows and festivals as Prophet *holidays*
so the deterministic spikes are learned explicitly.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Sequence, Tuple

from ..features.calendar_features import _FESTIVALS


def _india_holidays_frame(start: date, end: date):
    """Build a Prophet-style holidays DataFrame covering [start, end]."""
    import pandas as pd

    rows = []
    for year in range(start.year, end.year + 1):
        for (mm, dd), label in _FESTIVALS.items():
            try:
                d = date(year, mm, dd)
            except ValueError:
                continue
            rows.append({"holiday": label, "ds": pd.Timestamp(d),
                         "lower_window": 0, "upper_window": 1})
        # salary window: 1st-3rd and 28th-EOM as a recurring "holiday"
        for dd in (1, 2, 3):
            rows.append({"holiday": "salary_start", "ds": pd.Timestamp(date(year, 1, dd)),
                         "lower_window": 0, "upper_window": 0})
    return pd.DataFrame(rows)


class ProphetModel:
    def __init__(
        self,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        interval_width: float = 0.95,
        use_holidays: bool = True,
    ) -> None:
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.interval_width = interval_width
        self.use_holidays = use_holidays
        self._model = None
        self._last_date: date | None = None

    def fit(self, y: Sequence[float], dates: Sequence[date] | None = None) -> "ProphetModel":
        import pandas as pd
        from prophet import Prophet

        if dates is None:
            # assume a daily series starting at an arbitrary epoch
            dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(len(y))]
        df = pd.DataFrame({"ds": pd.to_datetime(list(dates)), "y": list(y)})
        self._last_date = pd.to_datetime(list(dates)[-1]).date()

        holidays = None
        if self.use_holidays:
            holidays = _india_holidays_frame(df["ds"].min().date(),
                                             df["ds"].max().date() + timedelta(days=400))

        self._model = Prophet(
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            interval_width=self.interval_width,
            holidays=holidays,
        )
        self._model.fit(df)
        return self

    def _future(self, h: int):
        return self._model.make_future_dataframe(periods=h, freq="D", include_history=False)

    def predict(self, h: int) -> List[float]:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        fc = self._model.predict(self._future(h))
        return list(map(float, fc["yhat"].tail(h)))

    def predict_interval(
        self, h: int, level: float = 0.95
    ) -> Tuple[List[float], List[float], List[float]]:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        fc = self._model.predict(self._future(h)).tail(h)
        return (
            list(map(float, fc["yhat"])),
            list(map(float, fc["yhat_lower"])),
            list(map(float, fc["yhat_upper"])),
        )

    @property
    def params(self) -> dict:
        return {
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_seasonality": self.yearly_seasonality,
            "interval_width": self.interval_width,
            "use_holidays": self.use_holidays,
        }
