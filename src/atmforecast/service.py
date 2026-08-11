r"""
Service layer: orchestrates data -> model -> forecast -> plan -> report.

Shared by the CLI, the FastAPI app and the batch pipeline so the business logic
lives in exactly one place. Pure-Python models are always available; library
models (SARIMA/Prophet/LSTM) are built on demand and only require their deps
when actually selected.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

from .data.loader import load_series
from .evaluation.backtest import compare_models, leaderboard
from .models.baselines import Drift, MeanForecast, MovingAverage, SeasonalNaive
from .models.holt_winters import HoltWinters
from .operations.cash_optimizer import recommend_cash_load
from .rag.report_builder import (
    build_backtest_report,
    build_cash_plan_report,
    build_forecast_report,
)


def build_model(name: str, season_length: int = 7):
    """Instantiate a forecaster by name. Library models imported lazily."""
    name = name.lower()
    if name == "mean":
        return MeanForecast()
    if name == "drift":
        return Drift()
    if name in ("moving_average", "ma"):
        return MovingAverage(window=season_length)
    if name in ("seasonal_naive", "snaive"):
        return SeasonalNaive(season_length=season_length)
    if name in ("holt_winters", "hw"):
        return HoltWinters(season_length=season_length, trend="add", seasonal="add")
    if name == "sarima":
        from .models.sarima import SarimaModel

        return SarimaModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, season_length))
    if name == "prophet":
        from .models.prophet_model import ProphetModel

        return ProphetModel()
    if name == "lstm":
        from .models.lstm_model import LSTMForecaster

        return LSTMForecaster()
    raise ValueError(f"unknown model: {name!r}")


# baselines available everywhere for the leaderboard
_BACKTEST_FACTORIES = {
    "mean": lambda: MeanForecast(),
    "drift": lambda: Drift(),
    "moving_average": lambda: MovingAverage(7),
    "seasonal_naive": lambda: SeasonalNaive(7),
    "holt_winters": lambda: HoltWinters(7, "add", "add"),
}


def forecast_atm(
    data_path: str,
    atm_id: str,
    model: str = "holt_winters",
    horizon: int = 14,
    target: str = "net_cash_out",
    season_length: int = 7,
    level: float = 0.95,
) -> Dict[str, object]:
    """Fit ``model`` on the full history of ``atm_id`` and forecast ``horizon`` days."""
    dates, y = load_series(data_path, atm_id, target)
    m = build_model(model, season_length)

    # library models with a date-aware signature
    if model.lower() == "prophet":
        m.fit(y, dates=dates)  # type: ignore[call-arg]
    else:
        m.fit(y)

    if hasattr(m, "predict_interval"):
        point, lower, upper = m.predict_interval(horizon, level=level)
    else:  # pragma: no cover
        point, lower, upper = m.predict(horizon), None, None

    start = dates[-1] + timedelta(days=1)
    forecast_dates = [start + timedelta(days=i) for i in range(horizon)]
    return {
        "atm_id": atm_id,
        "model": model,
        "horizon": horizon,
        "history_end": dates[-1].isoformat(),
        "forecast_start": start.isoformat(),
        "dates": [d.isoformat() for d in forecast_dates],
        "point": point,
        "lower": lower,
        "upper": upper,
        "level": level,
        "params": getattr(m, "params", {}),
        "residual_std": _residual_std(lower, upper, point, level),
    }


def _residual_std(lower, upper, point, level) -> Optional[float]:
    """Back out the 1-day residual std from a day-1 Gaussian interval."""
    from .models.base import z_for_level

    if not lower or not upper:
        return None
    z = z_for_level(level)
    if z == 0:
        return None
    return (upper[0] - lower[0]) / (2 * z)


def backtest_atm(
    data_path: str,
    atm_id: str,
    horizon: int = 14,
    target: str = "net_cash_out",
    min_train_size: int = 365,
    step: int = 14,
    season_length: int = 7,
) -> List[dict]:
    """Run the rolling-origin leaderboard over the pure-Python model suite."""
    _, y = load_series(data_path, atm_id, target)
    reports = compare_models(
        y, _BACKTEST_FACTORIES, horizon=horizon,
        min_train_size=min_train_size, step=step, seasonality=season_length,
    )
    return leaderboard(reports, sort_by="MASE")


def cash_plan_atm(
    forecast: Dict[str, object],
    service_level: float = 0.95,
    current_balance: Optional[float] = None,
) -> Dict[str, object]:
    """Turn a forecast dict into a cash-replenishment plan."""
    point = forecast["point"]  # type: ignore[index]
    resid_std = forecast.get("residual_std") or 0.0  # type: ignore[union-attr]
    plan = recommend_cash_load(point, resid_std, service_level=service_level)
    return {
        "atm_id": forecast["atm_id"],
        "service_level": plan.service_level,
        "cycle_load": plan.cycle_load,
        "per_day_load": plan.per_day_load,
        "safety_stock": plan.safety_stock,
        "expected_stockout_prob": plan.expected_stockout_prob,
        "current_balance": current_balance,
        "suggested_topup": (
            None if current_balance is None else max(0.0, plan.cycle_load - current_balance)
        ),
    }


def build_reports_for_atm(
    data_path: str,
    atm_id: str,
    model: str = "holt_winters",
    horizon: int = 14,
    service_level: float = 0.95,
    location_type: str = "",
) -> List[Dict[str, object]]:
    """Produce the three RAG documents (forecast, backtest, cash plan) for an ATM."""
    fc = forecast_atm(data_path, atm_id, model=model, horizon=horizon)
    board = backtest_atm(data_path, atm_id, horizon=horizon)
    plan = cash_plan_atm(fc, service_level=service_level)

    start = date.fromisoformat(fc["forecast_start"])  # type: ignore[arg-type]
    docs = [
        build_forecast_report(
            atm_id, model, start, fc["point"], fc["lower"], fc["upper"], location_type
        ),
        build_backtest_report(atm_id, board),
        build_cash_plan_report(
            atm_id, plan["service_level"], plan["per_day_load"], plan["cycle_load"],
            plan["safety_stock"], plan["expected_stockout_prob"], plan.get("current_balance"),
        ),
    ]
    return docs
