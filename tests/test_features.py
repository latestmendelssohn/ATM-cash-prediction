"""Tests for the pure-stdlib parts of the calendar feature engineering."""
import math
from datetime import date

from atmforecast.features.calendar_features import (
    date_features,
    festival_label,
    fourier_terms,
    future_dates,
    is_festival,
)


def test_weekend_and_weekday_flags():
    sat = date(2021, 1, 2)   # Saturday
    mon = date(2021, 1, 4)   # Monday
    assert date_features(sat)["is_weekend"] == 1.0
    assert date_features(mon)["is_monday"] == 1.0
    assert date_features(mon)["is_weekend"] == 0.0


def test_month_start_and_end_flags():
    assert date_features(date(2021, 3, 1))["is_month_start"] == 1.0
    assert date_features(date(2021, 3, 31))["is_month_end"] == 1.0   # 31 days
    assert date_features(date(2021, 2, 27))["is_month_end"] == 1.0   # Feb: 28-2
    assert date_features(date(2021, 3, 15))["is_month_start"] == 0.0


def test_salary_window_union():
    assert date_features(date(2021, 6, 2))["is_salary_window"] == 1.0
    assert date_features(date(2021, 6, 29))["is_salary_window"] == 1.0
    assert date_features(date(2021, 6, 12))["is_salary_window"] == 0.0


def test_festival_detection():
    assert is_festival(date(2021, 11, 12)) is True
    assert festival_label(date(2021, 11, 12)) == "diwali"
    assert is_festival(date(2021, 6, 6)) is False


def test_one_hot_dow_drops_monday():
    feats = date_features(date(2021, 1, 4))  # Monday
    assert all(feats[f"dow_{k}"] == 0.0 for k in range(1, 7))
    tue = date_features(date(2021, 1, 5))    # Tuesday -> dow_1
    assert tue["dow_1"] == 1.0


def test_fourier_period_7_repeats_weekly():
    t0 = date(2021, 1, 1)
    a = fourier_terms(date(2021, 1, 1), 7.0, 2, t0)
    b = fourier_terms(date(2021, 1, 8), 7.0, 2, t0)  # +7 days
    for k in a:
        assert math.isclose(a[k], b[k], abs_tol=1e-9)


def test_future_dates_length_and_order():
    fd = future_dates(date(2021, 1, 31), 3)
    assert fd == [date(2021, 2, 1), date(2021, 2, 2), date(2021, 2, 3)]
