"""Shared logic for the Streamlit and Gradio demos.

The UI files stay thin. This module is the only demo-specific layer between them
and the project's data/model core.
"""
from __future__ import annotations

from datetime import timedelta

import data
import models

DATA_PATH = data.DEFAULT_CSV
ATMS = data.list_atms(DATA_PATH)


def inr(value: float) -> str:
    value = float(value)
    if abs(value) >= 10_000_000:
        return f"Rs {value / 10_000_000:,.2f} Cr"
    if abs(value) >= 100_000:
        return f"Rs {value / 100_000:,.2f} L"
    return f"Rs {value:,.0f}"


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
            "forecast (Rs)": round(point[i]),
            "lower (Rs)": round(lower[i]),
            "upper (Rs)": round(upper[i]),
        }
        for i in range(horizon)
    ]
    summary = f"""### {atm_id}

This demo uses **Holt-Winters** on the bundled synthetic data.

- Forecast window: **{horizon} days**, starting **{start.isoformat()}**
- Forecast total: **{inr(sum(point))}**
- 95% daily interval: widens with the horizon
- Recommended cycle load: **{inr(plan['cycle_load'])}**
- Safety stock: **{inr(plan['safety_stock'])}**
- Measured cycle-error spread: **{inr(cycle_sigma)}**
- Target service level: **{service_level:.0%}**

The demo uses synthetic data only. It does not call Gemini and does not need an API key.
"""
    return summary, rows
