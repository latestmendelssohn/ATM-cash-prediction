"""Shared logic for the Streamlit and Gradio demos.

The UI files stay thin. This module is the only demo-specific layer between them
and the project's data/model core. It uses the small public 2009-2010 ATM
dataset processed to ``data/processed/atm_daily.csv``. The source CSV does not
identify a currency, so numbers here are formatted with thousand separators and
no currency symbol.
"""
from __future__ import annotations

from datetime import timedelta

import data
import models

DATA_PATH = data.DEFAULT_CSV
# ATM3 has almost only zero-withdrawal days in the source and is not a
# meaningful forecasting series, so it is left out of the demo choices.
ATMS = [atm for atm in data.list_atms(DATA_PATH) if atm != "ATM3"]


def fmt(value: float) -> str:
    """Format a number with thousand separators. The source has no stated unit."""
    return f"{float(value):,.0f}"


def run_forecast(atm_id: str, horizon: int, service_level: float):
    """Return a Markdown summary and daily rows for the selected ATM."""
    horizon = int(horizon)
    service_level = float(service_level)
    if atm_id not in ATMS:
        raise ValueError(f"unknown ATM: {atm_id}")
    if not 1 <= horizon <= 30:
        raise ValueError("horizon must be between 1 and 30 days")
    if not 0.80 <= service_level <= 0.99:
        raise ValueError("service level must be between 0.80 and 0.99")

    dates, history = data.load_series(DATA_PATH, atm_id)
    model = models.HoltWinters(7, "add", "add").fit(history)
    point, lower, upper = model.predict_interval(horizon, service_level)
    cycle_sigma = models.cycle_sigma_from_backtest(
        history, lambda: models.build_model("holt_winters"), horizon=horizon
    )
    plan = models.recommend_cash_load(
        point, model._sigma, service_level, cycle_sigma=cycle_sigma
    )
    start = dates[-1] + timedelta(days=1)
    rows = [
        {
            "date": (start + timedelta(days=i)).isoformat(),
            "forecast": round(point[i]),
            "lower": round(lower[i]),
            "upper": round(upper[i]),
        }
        for i in range(horizon)
    ]
    summary = f"""### {atm_id}

This demo uses **Holt-Winters** on the public 2009-2010 ATM dataset.
Values keep the source unit; the source CSV does not identify a currency.

- Forecast window: **{horizon} days**, starting **{start.isoformat()}**
- Forecast total: **{fmt(sum(point))}**
- 95% daily interval: widens with the horizon
- Recommended cycle load: **{fmt(plan['cycle_load'])}**
- Safety stock: **{fmt(plan['safety_stock'])}**
- Measured cycle-error spread: **{fmt(cycle_sigma)}**
- Target service level: **{service_level:.0%}**

The demo does not call Gemini and does not need an API key.
"""
    return summary, rows
