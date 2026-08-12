"""Run the existing forecasting leaderboard on the prepared real dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import data
import models

DEFAULT_MODELS = ("mean", "seasonal_naive", "holt_winters")


def selected_atms(path, excluded=()):
    excluded = set(excluded)
    return [atm for atm in data.list_atms(path) if atm not in excluded]


def run_backtest(path, atms, horizon=14, min_train=180, step=14):
    results = {}
    for atm_id in atms:
        _, values = data.load_series(path, atm_id)
        results[atm_id] = {
            "observations": len(values),
            "leaderboard": models.leaderboard(
                values,
                models=DEFAULT_MODELS,
                horizon=horizon,
                min_train=min_train,
                step=step,
            ),
        }
    return results


def render_report(results, horizon, min_train, step, excluded):
    lines = [
        "# Real-data forecast comparison",
        "",
        "This preliminary comparison reuses the project's existing rolling-origin backtester.",
        f"The horizon is {horizon} days, the minimum training window is {min_train} days, and the origin moves by {step} days.",
        "",
        f"ATM IDs excluded before modelling: {', '.join(excluded) if excluded else 'none'}.",
        "ATM3 is excluded because the EDA found 362 zero days out of 365.",
        "",
    ]
    for atm_id, result in results.items():
        lines.extend([
            f"## {atm_id}",
            "",
            f"Usable observations: {result['observations']}",
            "",
            "| model | MAE | RMSE | MAPE | sMAPE | MASE | folds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in result["leaderboard"]:
            lines.append(
                f"| {row['model']} | {row['MAE']:.2f} | {row['RMSE']:.2f} | "
                f"{row['MAPE']:.2f} | {row['sMAPE']:.2f} | {row['MASE']:.2f} | {row['folds']} |"
            )
        lines.extend(["", "Lower scores are better. MASE below 1 means the model beats the seasonal-naive scale.", ""])
    lines.extend([
        "## Limitations of this first run",
        "",
        "- The two missing ATM1 days and three missing ATM2 days were skipped by preprocessing, so the series is slightly shorter than the calendar range.",
        "- ATM4's 10,920 maximum was retained. It should be treated as a data-quality decision, not silently removed after seeing the scores.",
        "- MAPE is not reliable for ATM2 because zero withdrawals make percentage errors explode; MASE, MAE, RMSE, and sMAPE are more useful here.",
        "- The one-year source limits the training window and the number of rolling folds.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest models on prepared real ATM data")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--min-train", type=int, default=180)
    parser.add_argument("--step", type=int, default=14)
    args = parser.parse_args()
    atms = selected_atms(args.input, args.exclude)
    results = run_backtest(args.input, atms, args.horizon, args.min_train, args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_report(results, args.horizon, args.min_train, args.step, args.exclude),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
