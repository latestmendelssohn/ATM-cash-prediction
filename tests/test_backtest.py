"""Tests for the rolling-origin backtesting harness."""
from atmforecast.evaluation.backtest import (
    compare_models,
    leaderboard,
    rolling_origin_backtest,
)
from atmforecast.models.baselines import MeanForecast, SeasonalNaive


def _series():
    pattern = [10, -5, 0, 3, -2, 8, -14]
    return [100 + 0.2 * t + pattern[t % 7] for t in range(200)]


def test_number_of_folds_is_correct():
    y = _series()
    rep = rolling_origin_backtest(
        y, lambda: SeasonalNaive(7), "sn", horizon=14, min_train_size=120, step=7
    )
    # origins: 120, 127, ... while origin+14 <= 200  -> last origin 182
    expected = len(range(120, 200 - 14 + 1, 7))
    assert len(rep.folds) == expected
    assert all(set(f.metrics) >= {"MAE", "RMSE", "MAPE", "sMAPE"} for f in rep.folds)


def test_max_folds_caps_iterations():
    y = _series()
    rep = rolling_origin_backtest(
        y, lambda: MeanForecast(), "mean", horizon=7, min_train_size=100, step=5, max_folds=3
    )
    assert len(rep.folds) == 3


def test_seasonal_naive_beats_mean_on_seasonal_series():
    y = _series()
    reports = compare_models(
        y,
        {"seasonal_naive": lambda: SeasonalNaive(7), "mean": lambda: MeanForecast()},
        horizon=14,
        min_train_size=120,
        step=7,
        seasonality=7,
    )
    board = leaderboard(reports, sort_by="MASE")
    assert board[0]["model"] == "seasonal_naive"


def test_aggregate_and_dispersion_keys_match():
    y = _series()
    rep = rolling_origin_backtest(y, lambda: SeasonalNaive(7), "sn", horizon=7, min_train_size=100)
    assert set(rep.aggregate()) == set(rep.dispersion())
