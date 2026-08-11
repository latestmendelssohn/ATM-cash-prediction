r"""
Forecast-report builder  (pure standard library).
=================================================

The RAG analyst does not reason over raw numbers -- it reasons over *documents*.
This module renders the numerical artifacts produced by the pipeline (forecasts,
backtest leaderboards, cash-replenishment plans) into compact, self-describing
Markdown snippets. Each snippet is one retrievable chunk in ChromaDB, tagged
with metadata (atm_id, kind, horizon) so the retriever can filter precisely.

Keeping this pure-Python means the report text is deterministic and unit
testable, independent of the LLM/vector-store stack.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Sequence


def _inr(x: float) -> str:
    """Format a rupee amount in the Indian lakh/crore convention."""
    x = float(x)
    if abs(x) >= 1e7:
        return f"Rs {x/1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"Rs {x/1e5:,.2f} L"
    return f"Rs {x:,.0f}"


def build_forecast_report(
    atm_id: str,
    model_name: str,
    start_date: date,
    point: Sequence[float],
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    location_type: str = "",
) -> Dict[str, object]:
    """Render a per-ATM forecast into a Markdown document + metadata."""
    h = len(point)
    total = sum(point)
    peak_i = max(range(h), key=lambda i: point[i])
    trough_i = min(range(h), key=lambda i: point[i])

    lines = [
        f"# Cash-demand forecast for {atm_id}"
        + (f" ({location_type})" if location_type else ""),
        f"Model: **{model_name}**  |  Horizon: **{h} days** starting {start_date.isoformat()}.",
        "",
        f"- Total forecast cash demand over the horizon: **{_inr(total)}**.",
        f"- Average daily demand: **{_inr(total / h)}**.",
        f"- Peak demand day: day {peak_i + 1} at **{_inr(point[peak_i])}**.",
        f"- Lowest demand day: day {trough_i + 1} at **{_inr(point[trough_i])}**.",
    ]
    if lower is not None and upper is not None:
        band = sum(u - l for l, u in zip(lower, upper)) / h
        lines.append(
            f"- 95% prediction interval average width: **{_inr(band)}** "
            f"(day-1 range {_inr(lower[0])} to {_inr(upper[0])})."
        )
    lines.append("")
    lines.append("Daily forecast (INR):")
    for i in range(h):
        rng = ""
        if lower is not None and upper is not None:
            rng = f"  [{_inr(lower[i])}, {_inr(upper[i])}]"
        lines.append(f"  day {i + 1}: {_inr(point[i])}{rng}")

    text = "\n".join(lines)
    meta = {
        "kind": "forecast",
        "atm_id": atm_id,
        "model": model_name,
        "horizon": h,
        "start_date": start_date.isoformat(),
        "total_demand": round(total, 2),
    }
    return {"id": f"forecast::{atm_id}::{model_name}", "text": text, "metadata": meta}


def build_backtest_report(atm_id: str, leaderboard_rows: List[dict]) -> Dict[str, object]:
    """Render the model-comparison leaderboard into a document."""
    lines = [
        f"# Backtest leaderboard for {atm_id}",
        "Rolling-origin (walk-forward) cross-validation. Lower is better; "
        "MASE < 1 means the model beats the seasonal-naive benchmark.",
        "",
    ]
    if leaderboard_rows:
        cols = [c for c in leaderboard_rows[0].keys()]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for r in leaderboard_rows:
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        best = leaderboard_rows[0]
        lines += [
            "",
            f"**Best model: {best.get('model')}** "
            f"(MASE {best.get('MASE')}, MAPE {best.get('MAPE')}%).",
        ]
    text = "\n".join(lines)
    meta = {
        "kind": "backtest",
        "atm_id": atm_id,
        "best_model": leaderboard_rows[0]["model"] if leaderboard_rows else "",
    }
    return {"id": f"backtest::{atm_id}", "text": text, "metadata": meta}


def build_cash_plan_report(
    atm_id: str,
    service_level: float,
    per_day_load: Sequence[float],
    cycle_load: float,
    safety_stock: float,
    expected_stockout_prob: float,
    current_balance: Optional[float] = None,
) -> Dict[str, object]:
    """Render the recommended cash-replenishment plan into a document."""
    h = len(per_day_load)
    lines = [
        f"# Cash-replenishment recommendation for {atm_id}",
        f"Target service level: **{service_level:.0%}** "
        f"(residual stock-out probability {expected_stockout_prob:.2%}).",
        "",
        f"- Recommended cash to load for the {h}-day cycle: **{_inr(cycle_load)}**.",
        f"- Of which safety stock (uncertainty buffer): **{_inr(safety_stock)}**.",
    ]
    if current_balance is not None:
        top_up = max(0.0, cycle_load - current_balance)
        lines.append(f"- Current balance: **{_inr(current_balance)}**.")
        lines.append(f"- Suggested top-up now: **{_inr(top_up)}**.")
    lines.append("")
    lines.append("Suggested per-day base-stock levels (INR):")
    for i, v in enumerate(per_day_load, start=1):
        lines.append(f"  day {i}: {_inr(v)}")

    text = "\n".join(lines)
    meta = {
        "kind": "cash_plan",
        "atm_id": atm_id,
        "service_level": service_level,
        "cycle_load": round(cycle_load, 2),
        "stockout_prob": round(expected_stockout_prob, 4),
    }
    return {"id": f"cashplan::{atm_id}", "text": text, "metadata": meta}
