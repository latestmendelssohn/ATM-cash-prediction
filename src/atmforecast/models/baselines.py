r"""
Baseline forecasters (pure standard library).

These cheap benchmarks are the yardstick every serious model must beat -- a
SARIMA that cannot outperform *seasonal naive* is not worth deploying. Each
model also exposes ``predict_interval`` via one-step in-sample residuals so it
can feed the cash-inventory optimiser.

Models
------
MeanForecast    : \hat y_{n+k} = \bar y                      (historical mean)
Drift           : \hat y_{n+k} = y_n + k\,(y_n-y_1)/(n-1)    (line through 1st & last)
MovingAverage   : \hat y_{n+k} = mean of last ``window`` obs (flat)
SeasonalNaive   : \hat y_{n+k} = y_{n+k-m}                   (repeat last season)
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .base import interval_from_residuals, mean, std


class _FittedMixin:
    """Provides residual-based prediction intervals to the baselines."""

    _resid_std: float = 0.0

    def _one_step_residuals(self, y: Sequence[float]) -> List[float]:  # pragma: no cover
        raise NotImplementedError

    def _set_residual_std(self, y: Sequence[float]) -> None:
        res = self._one_step_residuals(y)
        self._resid_std = std(res) if len(res) > 1 else 0.0

    def predict_interval(self, h: int, level: float = 0.95) -> Tuple[List[float], List[float], List[float]]:
        point = self.predict(h)  # type: ignore[attr-defined]
        lower, upper = interval_from_residuals(point, self._resid_std, level)
        return point, lower, upper


class MeanForecast(_FittedMixin):
    """Forecast every future value as the historical mean."""

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, y: Sequence[float]) -> "MeanForecast":
        self._y = list(y)
        self._mean = mean(self._y)
        self._set_residual_std(self._y)
        return self

    def _one_step_residuals(self, y: Sequence[float]) -> List[float]:
        # expanding-mean one-step-ahead residuals
        res = []
        run = 0.0
        for i in range(1, len(y)):
            run = sum(y[:i]) / i
            res.append(y[i] - run)
        return res

    def predict(self, h: int) -> List[float]:
        return [self._mean] * h


class Drift(_FittedMixin):
    r"""Random walk with drift: extrapolate the line through the first and last point."""

    def __init__(self) -> None:
        self._last = 0.0
        self._slope = 0.0

    def fit(self, y: Sequence[float]) -> "Drift":
        self._y = list(y)
        n = len(self._y)
        self._last = self._y[-1]
        self._slope = (self._y[-1] - self._y[0]) / (n - 1) if n > 1 else 0.0
        self._set_residual_std(self._y)
        return self

    def _one_step_residuals(self, y: Sequence[float]) -> List[float]:
        # naive one-step: yhat_t = y_{t-1} + slope
        res = []
        for i in range(1, len(y)):
            slope = (y[i - 1] - y[0]) / i if i >= 1 else 0.0
            res.append(y[i] - (y[i - 1] + slope))
        return res

    def predict(self, h: int) -> List[float]:
        return [self._last + self._slope * k for k in range(1, h + 1)]


class MovingAverage(_FittedMixin):
    """Flat forecast equal to the mean of the last ``window`` observations."""

    def __init__(self, window: int = 7) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._level = 0.0

    def fit(self, y: Sequence[float]) -> "MovingAverage":
        self._y = list(y)
        w = min(self.window, len(self._y))
        self._level = mean(self._y[-w:])
        self._set_residual_std(self._y)
        return self

    def _one_step_residuals(self, y: Sequence[float]) -> List[float]:
        res = []
        for i in range(self.window, len(y)):
            pred = mean(y[i - self.window:i])
            res.append(y[i] - pred)
        return res

    def predict(self, h: int) -> List[float]:
        return [self._level] * h


class SeasonalNaive(_FittedMixin):
    r"""Repeat the last observed season: :math:`\hat y_{n+k}=y_{n+k-m}`."""

    def __init__(self, season_length: int = 7) -> None:
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.m = season_length

    def fit(self, y: Sequence[float]) -> "SeasonalNaive":
        self._y = list(y)
        if len(self._y) < self.m:
            raise ValueError("series shorter than one season")
        self._last_season = self._y[-self.m:]
        self._set_residual_std(self._y)
        return self

    def _one_step_residuals(self, y: Sequence[float]) -> List[float]:
        return [y[t] - y[t - self.m] for t in range(self.m, len(y))]

    def predict(self, h: int) -> List[float]:
        # tile the last season across the horizon
        return [self._last_season[k % self.m] for k in range(h)]
