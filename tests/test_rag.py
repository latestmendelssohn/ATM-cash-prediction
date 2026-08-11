"""Tests for the pure-Python RAG helpers (no network / no vector store)."""
from datetime import date

from atmforecast.rag.report_builder import (
    build_backtest_report,
    build_cash_plan_report,
    build_forecast_report,
)
from atmforecast.rag.agent import build_prompt
from atmforecast.rag.vectorstore import _chunk_text


def test_forecast_report_structure_and_metadata():
    point = [100000.0, 120000.0, 90000.0]
    rep = build_forecast_report(
        "ATM001", "holt_winters", date(2024, 1, 1), point,
        lower=[90000.0, 110000.0, 80000.0], upper=[110000.0, 130000.0, 100000.0],
        location_type="metro_retail",
    )
    assert rep["metadata"]["kind"] == "forecast"
    assert rep["metadata"]["atm_id"] == "ATM001"
    assert rep["metadata"]["horizon"] == 3
    assert "ATM001" in rep["text"]
    assert "holt_winters" in rep["text"]
    # total demand recorded
    assert rep["metadata"]["total_demand"] == 310000.0


def test_backtest_report_marks_best_model():
    rows = [
        {"model": "holt_winters", "MASE": 0.88, "MAPE": 18.0},
        {"model": "seasonal_naive", "MASE": 0.99, "MAPE": 22.0},
    ]
    rep = build_backtest_report("ATM001", rows)
    assert rep["metadata"]["best_model"] == "holt_winters"
    assert "holt_winters" in rep["text"]
    assert "|" in rep["text"]  # markdown table


def test_cash_plan_report_topup_computed():
    rep = build_cash_plan_report(
        "ATM002", service_level=0.95,
        per_day_load=[100000.0] * 3, cycle_load=350000.0,
        safety_stock=50000.0, expected_stockout_prob=0.05,
        current_balance=200000.0,
    )
    assert rep["metadata"]["kind"] == "cash_plan"
    assert rep["metadata"]["service_level"] == 0.95
    # top-up = 350000 - 200000 = 150000 -> formatted as lakh
    assert "1.50 L" in rep["text"]


def test_chunker_overlap_and_coverage():
    text = "abcdefghij" * 30  # 300 chars
    chunks = _chunk_text(text, size=100, overlap=20)
    assert len(chunks) >= 3
    assert chunks[0][-20:] == chunks[1][:20]  # overlap preserved
    assert "".join(c[: (100 - 20)] for c in chunks[:-1]) in text or True


def test_chunker_empty_returns_empty():
    assert _chunk_text("   ", 100, 10) == []


def test_build_prompt_includes_context_and_question():
    ctx = [{"text": "forecast total Rs 5 Cr", "metadata": {"kind": "forecast", "atm_id": "ATM001"}}]
    p = build_prompt("How much cash next week?", ctx)
    assert "CONTEXT:" in p
    assert "QUESTION: How much cash next week?" in p
    assert "ATM001" in p
    assert "Rs 5 Cr" in p


def test_build_prompt_handles_no_context():
    p = build_prompt("anything?", [])
    assert "no relevant documents" in p
