r"""
Calendar & holiday feature engineering.
=======================================

ATM demand is driven far more by the *calendar* than by its own recent history:
salaries land at month start/end, rents are paid on the 1st, and festivals
create large but predictable spikes. Pure autoregressive models (ARIMA,
Holt-Winters) cannot see these events, so we encode them as **exogenous
regressors** ``X`` and feed them to SARIMAX, Prophet and the LSTM.

The per-date encoder ``date_features`` is pure standard library (and unit
tested). ``build_design_matrix`` assembles a pandas DataFrame aligned to a
DatetimeIndex for the library models. Weekly/annual cycles are additionally
represented with **Fourier terms**

    .. math:: \sin\!\Big(\tfrac{2\pi k t}{P}\Big),\; \cos\!\Big(\tfrac{2\pi k t}{P}\Big)

which give a smooth, low-parameter basis for seasonality of period ``P``.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, List, Sequence

# Salary-cycle windows (kept consistent with the data generator).
_MONTH_START_DAYS = {1, 2, 3}
_MID_MONTH_DAYS = {7, 15}

# Approximate Indian festival / public-holiday calendar (month, day) -> label.
# In production, swap this for the `holidays` package (holidays.India()).
_FESTIVALS = {
    (1, 1): "new_year",
    (1, 26): "republic_day",
    (3, 8): "holi",
    (4, 14): "baisakhi",
    (8, 15): "independence_day",
    (8, 30): "festival_season",
    (10, 2): "gandhi_jayanti",
    (10, 24): "dhanteras",
    (11, 12): "diwali",
    (12, 25): "christmas",
    (12, 31): "new_year_eve",
}


def _days_in_month(d: date) -> int:
    nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return (nxt - timedelta(days=1)).day


def is_festival(d: date) -> bool:
    return (d.month, d.day) in _FESTIVALS


def festival_label(d: date) -> str:
    return _FESTIVALS.get((d.month, d.day), "")


def date_features(d: date) -> Dict[str, float]:
    """Return a dict of calendar features for a single date (pure stdlib)."""
    dow = d.weekday()  # Mon=0 .. Sun=6
    dim = _days_in_month(d)
    dom = d.day

    feats: Dict[str, float] = {
        "dow": float(dow),
        "is_weekend": 1.0 if dow >= 5 else 0.0,
        "is_monday": 1.0 if dow == 0 else 0.0,
        "is_friday": 1.0 if dow == 4 else 0.0,
        "day_of_month": float(dom),
        "is_month_start": 1.0 if dom in _MONTH_START_DAYS else 0.0,
        "is_month_end": 1.0 if dom >= dim - 2 else 0.0,
        "is_mid_month": 1.0 if dom in _MID_MONTH_DAYS else 0.0,
        "is_salary_window": 1.0
        if (dom in _MONTH_START_DAYS or dom >= dim - 2)
        else 0.0,
        "is_festival": 1.0 if is_festival(d) else 0.0,
        "month": float(d.month),
        "quarter": float((d.month - 1) // 3 + 1),
    }
    # one-hot day-of-week (drop Monday to avoid the dummy-variable trap)
    for k in range(1, 7):
        feats[f"dow_{k}"] = 1.0 if dow == k else 0.0
    return feats


def fourier_terms(d: date, period: float, order: int, t0: date) -> Dict[str, float]:
    """Fourier basis of a given period for smooth seasonality."""
    t = (d - t0).days
    out: Dict[str, float] = {}
    for k in range(1, order + 1):
        ang = 2.0 * math.pi * k * t / period
        out[f"sin_{int(period)}_{k}"] = math.sin(ang)
        out[f"cos_{int(period)}_{k}"] = math.cos(ang)
    return out


def feature_names(
    weekly_fourier: int = 3, yearly_fourier: int = 4, t0: date | None = None
) -> List[str]:
    """Column order produced by :func:`build_design_matrix`."""
    ref = t0 or date(2020, 1, 1)
    names = list(date_features(ref).keys())
    names += list(fourier_terms(ref, 7.0, weekly_fourier, ref).keys())
    names += list(fourier_terms(ref, 365.25, yearly_fourier, ref).keys())
    return names


def _row_for_date(
    d: date, t0: date, weekly_fourier: int, yearly_fourier: int
) -> Dict[str, float]:
    row = date_features(d)
    row.update(fourier_terms(d, 7.0, weekly_fourier, t0))
    row.update(fourier_terms(d, 365.25, yearly_fourier, t0))
    return row


def build_design_matrix(
    dates: Sequence[date],
    weekly_fourier: int = 3,
    yearly_fourier: int = 4,
):
    """Assemble a pandas DataFrame of exogenous regressors indexed by date.

    Requires pandas (used by the library models). The reference epoch ``t0``
    for the Fourier terms is the first date, so train/future matrices stay
    phase-aligned as long as the same ``t0`` is used.
    """
    import pandas as pd

    if len(dates) == 0:
        raise ValueError("dates is empty")
    t0 = dates[0]
    rows = [_row_for_date(d, t0, weekly_fourier, yearly_fourier) for d in dates]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))
    return df


def future_dates(last: date, horizon: int) -> List[date]:
    """The next ``horizon`` daily dates after ``last`` (for out-of-sample X)."""
    return [last + timedelta(days=k) for k in range(1, horizon + 1)]
