#!/usr/bin/env python3
"""
End-to-end batch pipeline (pure-Python core; no third-party deps required).

Steps
-----
1. (optional) generate the synthetic dataset if it is missing
2. for every ATM: rolling-origin backtest -> pick best model -> forecast
   -> compute the 95% cash-replenishment plan
3. write machine-readable artifacts to ``artifacts/`` and print a summary

The artifacts (forecasts.json, leaderboards.json, cash_plans.json,
reports.jsonl) are exactly what the RAG layer embeds. Run:

    python scripts/run_pipeline.py --horizon 14 --service-level 0.95
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# make ``src`` importable when run directly (no install needed)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atmforecast import service                          # noqa: E402
from atmforecast.data.generate_synthetic import generate  # noqa: E402
from atmforecast.data.loader import list_atms            # noqa: E402


def _location_map(data_path: str) -> dict:
    """Best-effort atm_id -> location_type map using stdlib csv."""
    import csv

    out = {}
    with open(data_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["atm_id"], row.get("location_type", ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full ATM forecasting pipeline.")
    ap.add_argument("--data", default="data/raw/atm_transactions.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--service-level", dest="service_level", type=float, default=0.95)
    ap.add_argument("--index", action="store_true", help="also embed reports into ChromaDB")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[pipeline] dataset not found -> generating {data_path}")
        from datetime import date

        generate(out_path=data_path, n_atms=5, start=date(2021, 1, 1), n_days=3 * 365, seed=42)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loc_map = _location_map(str(data_path))
    atms = list_atms(str(data_path))
    print(f"[pipeline] {len(atms)} ATMs: {atms}")

    forecasts, leaderboards, cash_plans, reports = {}, {}, {}, []

    for atm in atms:
        board = service.backtest_atm(str(data_path), atm, horizon=args.horizon)
        best = board[0]["model"] if board else "holt_winters"
        leaderboards[atm] = board

        fc = service.forecast_atm(str(data_path), atm, model=best, horizon=args.horizon)
        forecasts[atm] = fc

        plan = service.cash_plan_atm(fc, service_level=args.service_level)
        cash_plans[atm] = plan

        docs = service.build_reports_for_atm(
            str(data_path), atm, model=best, horizon=args.horizon,
            service_level=args.service_level, location_type=loc_map.get(atm, ""),
        )
        reports.extend(docs)

        print(f"  {atm:<8} best={best:<16} "
              f"total14d={sum(fc['point']):>14,.0f}  "
              f"cycle_load={plan['cycle_load']:>14,.0f}  "
              f"stockout_p={plan['expected_stockout_prob']:.2%}")

    (out_dir / "forecasts.json").write_text(json.dumps(forecasts, indent=2, default=str))
    (out_dir / "leaderboards.json").write_text(json.dumps(leaderboards, indent=2, default=str))
    (out_dir / "cash_plans.json").write_text(json.dumps(cash_plans, indent=2, default=str))
    with open(out_dir / "reports.jsonl", "w", encoding="utf-8") as fh:
        for r in reports:
            fh.write(json.dumps(r, default=str) + "\n")

    print(f"[pipeline] wrote artifacts to {out_dir}/ "
          f"(forecasts, leaderboards, cash_plans, reports.jsonl)")

    if args.index:
        print("[pipeline] embedding reports into ChromaDB ...")
        from atmforecast.rag.agent import RAGAnalyst

        analyst = RAGAnalyst()
        added = analyst.store.add_reports(reports)
        print(f"[pipeline] indexed {added} documents (store total: {analyst.store.count()})")


if __name__ == "__main__":
    main()
