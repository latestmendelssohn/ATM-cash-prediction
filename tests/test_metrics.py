"""Unit tests for the pure-Python evaluation metrics."""
import math

import pytest

from atmforecast.evaluation import metrics as M


def test_perfect_forecast_is_zero_error():
    y = [10.0, 20.0, 30.0, 40.0]
    assert M.mae(y, y) == 0.0
    assert M.rmse(y, y) == 0.0
    assert M.mape(y, y) == 0.0
    assert M.smape(y, y) == 0.0


def test_mae_and_rmse_known_values():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 5.0]           # errors: 1, 0, 2
    assert M.mae(y_true, y_pred) == pytest.approx(1.0)          # (1+0+2)/3
    assert M.rmse(y_true, y_pred) == pytest.approx(math.sqrt(5 / 3))  # (1+0+4)/3


def test_rmse_ge_mae():
    y_true = [3.0, -1.0, 7.0, 2.5]
    y_pred = [2.0, 0.0, 5.0, 3.0]
    assert M.rmse(y_true, y_pred) >= M.mae(y_true, y_pred)


def test_mape_known_value():
    y_true = [100.0, 200.0]
    y_pred = [110.0, 180.0]            # 10% and 10%
    assert M.mape(y_true, y_pred) == pytest.approx(10.0)


def test_smape_is_symmetric_and_bounded():
    a, b = [100.0], [50.0]
    # sMAPE(a,b) uses |a|+|b| in the denominator -> symmetric in its arguments
    assert M.smape(a, b) == pytest.approx(M.smape(b, a))
    assert 0.0 <= M.smape([1.0], [1000.0]) <= 200.0


def test_mase_zero_for_perfect_forecast():
    # weekly-seasonal training series with a mild trend so the seasonal-naive
    # scaler is strictly positive (avoids a degenerate 0/0).
    base = [10, 12, 14, 9, 8, 11, 13]
    train = [v + 0.5 * w for w in range(4) for v in base]
    y_true = [10.0, 12.0, 14.0]
    y_pred = [10.0, 12.0, 14.0]        # perfect -> MAE 0 -> MASE 0
    assert M.mase(y_true, y_pred, train, seasonality=7) == pytest.approx(0.0)


def test_mase_infinite_when_scaler_degenerate():
    # perfectly periodic training -> seasonal-naive error is 0 -> scaler 0.
    train = [10, 12, 14, 9, 8, 11, 13] * 4
    assert M.mase([1.0], [2.0], train, seasonality=7) == float("inf")


def test_mase_equal_to_one_for_naive_like_error():
    train = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0]  # seasonal(2) diff abs = 0 -> inf scale guard
    # use seasonality 1: abs diffs are all 2 -> scale = 2
    train2 = [0.0, 2.0, 4.0, 6.0]            # |diff|=2 each -> scale 2
    y_true = [10.0, 12.0]
    y_pred = [12.0, 14.0]                     # abs error 2 each -> MAE 2 -> MASE 1
    assert M.mase(y_true, y_pred, train2, seasonality=1) == pytest.approx(1.0)


def test_coverage():
    y = [1.0, 2.0, 3.0, 4.0]
    lo = [0.0, 0.0, 0.0, 10.0]
    hi = [2.0, 2.0, 2.0, 20.0]         # inside: 1,2, not 3, not 4 -> 2/4
    assert M.coverage(y, lo, hi) == pytest.approx(0.5)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        M.mae([1.0, 2.0], [1.0])


def test_all_metrics_bundle_keys():
    train = [10, 12, 14, 9, 8, 11, 13] * 4
    out = M.all_metrics([10.0, 12.0], [11.0, 12.0], y_train=train, seasonality=7)
    assert set(out) == {"MAE", "RMSE", "MAPE", "sMAPE", "MASE"}
