"""Evaluate prediction-interval coverage using rolling-origin folds."""
from __future__ import annotations

import argparse
from pathlib import Path

import data
import models


def interval_metrics(y, factory, horizon=14, min_train=180, step=14, level=0.95):
    y = list(y)
    if len(y) < min_train + horizon:
        raise ValueError("series too short for interval evaluation")
    covered = [0] * horizon
    widths = [[] for _ in range(horizon)]
    total = [0] * horizon
    origin = min_train
    folds = 0
    while origin + horizon <= len(y):
        model = factory()
        model.fit(y[:origin])
        _, lower, upper = model.predict_interval(horizon, level)
        for index, actual in enumerate(y[origin:origin + horizon]):
            covered[index] += lower[index] <= actual <= upper[index]
            widths[index].append(upper[index] - lower[index])
            total[index] += 1
        folds += 1
        origin += step
    by_horizon = [
        {
            "horizon": index + 1,
            "coverage": covered[index] / total[index],
            "mean_width": sum(widths[index]) / len(widths[index]),
        }
        for index in range(horizon)
    ]
    all_widths = [width for values in widths for width in values]
    return {
        "folds": folds,
        "coverage": sum(covered) / sum(total),
        "mean_width": sum(all_widths) / len(all_widths),
        "by_horizon": by_horizon,
    }


def run_evaluation(path, atms, horizon=14, min_train=180, step=14, level=0.95):
    results = {}
    for atm_id in atms:
        _, values = data.load_series(path, atm_id)
        results[atm_id] = interval_metrics(
            values, lambda: models.build_model("holt_winters"),
            horizon, min_train, step, level,
        )
    return results


def render_report(results, horizon, min_train, step, level, excluded):
    lines = [
        "# Real-data interval coverage",
        "",
        f"Model: Holt-Winters. Nominal coverage: {level:.0%}. Horizon: {horizon} days. "
        f"Minimum training window: {min_train} days. Origin step: {step} days.",
        "",
        f"ATM IDs excluded before evaluation: {', '.join(excluded) if excluded else 'none'}.",
        "",
        "| ATM | Folds | Empirical coverage | Mean interval width |",
        "| --- | ---: | ---: | ---: |",
    ]
    for atm_id, result in results.items():
        lines.append(
            f"| {atm_id} | {result['folds']} | {result['coverage']:.2%} | "
            f"{result['mean_width']:.2f} |"
        )
    lines.extend(["", "## Coverage by forecast day", ""])
    for atm_id, result in results.items():
        lines.extend([f"### {atm_id}", "", "| Day | Coverage | Mean width |", "| ---: | ---: | ---: |"])
        for row in result["by_horizon"]:
            lines.append(f"| {row['horizon']} | {row['coverage']:.2%} | {row['mean_width']:.2f} |")
        lines.append("")
    lines.extend([
        "## Interpretation notes",
        "",
        "- Coverage near the nominal level is desirable, but wider intervals are not automatically better.",
        "- ATM4's 10,920 observation is retained. It can widen the residual estimate and affect coverage.",
        "- ATM3 was excluded because the EDA found 362 zero days out of 365.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ATM prediction intervals")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--min-train", type=int, default=180)
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--level", type=float, default=0.95)
    args = parser.parse_args()
    atms = [atm for atm in data.list_atms(args.input) if atm not in args.exclude]
    results = run_evaluation(args.input, atms, args.horizon, args.min_train, args.step, args.level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_report(results, args.horizon, args.min_train, args.step, args.level, args.exclude),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
