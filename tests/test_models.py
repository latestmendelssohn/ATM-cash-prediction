"""Unit tests for the pure-Python forecasting models."""
import math

import pytest

from atmforecast.models.baselines import Drift, MeanForecast, MovingAverage, SeasonalNaive
from atmforecast.models.holt_winters import HoltWinters


# ------------------------------- baselines ---------------------------------

def test_seasonal_naive_repeats_last_season():
    y = [1, 2, 3, 4, 5, 6, 7] * 3  # m=7
    m = SeasonalNaive(season_length=7).fit(y)
    fc = m.predict(10)
    assert fc[:7] == [1, 2, 3, 4, 5, 6, 7]
    assert fc[7:10] == [1, 2, 3]     # wraps around


def test_mean_forecast_is_flat_at_mean():
    y = [2.0, 4.0, 6.0]
    fc = MeanForecast().fit(y).predict(3)
    assert fc == [4.0, 4.0, 4.0]


def test_moving_average_uses_last_window():
    y = [10, 20, 30, 40, 50]
    fc = MovingAverage(window=3).fit(y).predict(2)
    assert fc == [40.0, 40.0]        # mean(30,40,50)


def test_drift_extrapolates_linear_trend():
    y = [0.0, 2.0, 4.0, 6.0, 8.0]    # perfect slope 2
    fc = Drift().fit(y).predict(3)
    assert fc == pytest.approx([10.0, 12.0, 14.0])


def test_seasonal_naive_needs_full_season():
    with pytest.raises(ValueError):
        SeasonalNaive(season_length=7).fit([1, 2, 3])


# ------------------------------ Holt-Winters -------------------------------

def test_hw_recovers_pure_linear_trend_no_season():
    y = [float(t) for t in range(30)]        # y_t = t
    hw = HoltWinters(season_length=7, trend="add", seasonal=None, optimize=True).fit(y)
    fc = hw.predict(5)
    # next values should be ~30,31,32,33,34
    for k, v in enumerate(fc, start=30):
        assert abs(v - k) < 1.0


def test_hw_additive_recovers_seasonal_pattern():
    # level 100, additive weekly pattern, no noise
    pattern = [10, -5, 0, 3, -2, 8, -14]     # sums to 0
    y = [100 + pattern[t % 7] for t in range(70)]
    hw = HoltWinters(season_length=7, trend=None, seasonal="add", optimize=True).fit(y)
    fc = hw.predict(7)
    for k in range(7):
        assert abs(fc[k] - (100 + pattern[k % 7])) < 3.0


def test_hw_multiplicative_positive_forecasts():
    pattern = [1.2, 0.8, 1.0, 1.1, 0.9, 1.3, 0.7]
    y = [200 * pattern[t % 7] for t in range(70)]
    hw = HoltWinters(season_length=7, trend=None, seasonal="mul", optimize=True).fit(y)
    fc = hw.predict(7)
    assert all(v > 0 for v in fc)
    for k in range(7):
        assert abs(fc[k] - 200 * pattern[k % 7]) < 15.0


def test_hw_interval_contains_point_and_widens():
    y = [100 + 2 * t + 5 * math.sin(t) for t in range(60)]
    hw = HoltWinters(season_length=7, trend="add", seasonal="add").fit(y)
    point, lower, upper = hw.predict_interval(10, level=0.95)
    assert all(lo <= p <= hi for lo, p, hi in zip(lower, point, upper))
    # random-walk-style bands widen with horizon
    assert (upper[-1] - lower[-1]) >= (upper[0] - lower[0])


def test_hw_params_in_unit_interval():
    y = [50 + (t % 7) + 0.1 * t for t in range(60)]
    hw = HoltWinters(season_length=7).fit(y)
    p = hw.params
    for name in ("alpha", "beta", "gamma"):
        assert 0.0 <= p[name] <= 1.0
