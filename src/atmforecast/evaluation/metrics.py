r"""
Point-forecast accuracy metrics  (pure standard library).

Notation
--------
Let :math:`y_1,\dots,y_h` be the actuals over the forecast horizon and
:math:`\hat y_1,\dots,\hat y_h` the forecasts.  We implement:

* **MAE**   -- Mean Absolute Error

  .. math:: \mathrm{MAE} = \frac{1}{h}\sum_{t=1}^{h}\lvert y_t-\hat y_t\rvert

* **RMSE**  -- Root Mean Squared Error

  .. math:: \mathrm{RMSE} = \sqrt{\frac{1}{h}\sum_{t=1}^{h}(y_t-\hat y_t)^2}

* **MAPE**  -- Mean Absolute Percentage Error (scale-free, undefined at y=0)

  .. math:: \mathrm{MAPE} = \frac{100}{h}\sum_{t=1}^{h}\frac{\lvert y_t-\hat y_t\rvert}{\lvert y_t\rvert}

* **sMAPE** -- symmetric MAPE, bounded in [0, 200]

  .. math:: \mathrm{sMAPE} = \frac{100}{h}\sum_{t=1}^{h}
            \frac{2\lvert y_t-\hat y_t\rvert}{\lvert y_t\rvert+\lvert\hat y_t\rvert}

* **MASE**  -- Mean Absolute Scaled Error (Hyndman & Koehler 2006). The forecast
  error is scaled by the in-sample MAE of a **seasonal naive** benchmark with
  season length :math:`m`. ``MASE < 1`` means we beat that benchmark.

  .. math:: \mathrm{MASE} = \frac{\frac1h\sum_t\lvert y_t-\hat y_t\rvert}
            {\frac{1}{n-m}\sum_{t=m+1}^{n}\lvert y^{tr}_t-y^{tr}_{t-m}\rvert}

* **coverage** -- empirical coverage of a prediction interval: the fraction of
  actuals that fall inside ``[lower, upper]``. Should be close to the nominal
  level (e.g. 0.95).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

Number = float


def _check(y_true: Sequence[Number], y_pred: Sequence[Number]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} != {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("empty input")


def mae(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    _check(y_true, y_pred)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: Sequence[Number], y_pred: Sequence[Number]) -> float:
    _check(y_true, y_pred)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def mape(y_true: Sequence[Number], y_pred: Sequence[Number], eps: float = 1e-8) -> float:
    _check(y_true, y_pred)
    return 100.0 * sum(
        abs(a - b) / max(abs(a), eps) for a, b in zip(y_true, y_pred)
    ) / len(y_true)


def smape(y_true: Sequence[Number], y_pred: Sequence[Number], eps: float = 1e-8) -> float:
    _check(y_true, y_pred)
    total = 0.0
    for a, b in zip(y_true, y_pred):
        denom = abs(a) + abs(b)
        total += 0.0 if denom < eps else 2.0 * abs(a - b) / denom
    return 100.0 * total / len(y_true)


def mase(
    y_true: Sequence[Number],
    y_pred: Sequence[Number],
    y_train: Sequence[Number],
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error using a seasonal-naive in-sample scaler."""
    _check(y_true, y_pred)
    if len(y_train) <= seasonality:
        raise ValueError("training series too short for the chosen seasonality")
    scale = sum(
        abs(y_train[t] - y_train[t - seasonality])
        for t in range(seasonality, len(y_train))
    ) / (len(y_train) - seasonality)
    if scale == 0:
        return float("inf")
    return mae(y_true, y_pred) / scale


def coverage(
    y_true: Sequence[Number],
    lower: Sequence[Number],
    upper: Sequence[Number],
) -> float:
    """Fraction of actuals inside the [lower, upper] prediction interval."""
    _check(y_true, lower)
    _check(y_true, upper)
    inside = sum(1 for a, lo, hi in zip(y_true, lower, upper) if lo <= a <= hi)
    return inside / len(y_true)


def all_metrics(
    y_true: Sequence[Number],
    y_pred: Sequence[Number],
    y_train: Sequence[Number] | None = None,
    seasonality: int = 7,
) -> Dict[str, float]:
    """Convenience bundle. MASE is only included when a training series is given."""
    out: Dict[str, float] = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }
    if y_train is not None:
        try:
            out["MASE"] = mase(y_true, y_pred, y_train, seasonality)
        except ValueError:
            out["MASE"] = float("nan")
    return out
