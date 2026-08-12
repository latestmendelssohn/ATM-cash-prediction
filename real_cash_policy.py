"""Apply the existing cash policy to real rolling-origin forecast errors."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import stdev

import data
import models

SERVICE_LEVELS = (0.90, 0.95, 0.975, 0.99)


def cycle_errors(y, factory, horizon=14, min_train=180, step=14):
    y = list(y)
    if len(y) < min_train + horizon:
        raise ValueError("series too short for cycle-error evaluation")
    errors = []
    origin = min_train
    while origin + horizon <= len(y):
        model = factory()
        model.fit(y[:origin])
        errors.append(sum(y[origin:origin + horizon]) - sum(model.predict(horizon)))
        origin += step
    return errors


def plan_for_atm(y, horizon=14, min_train=180, step=14):
    errors = cycle_errors(
        y, lambda: models.build_model("holt_winters"), horizon, min_train, step
    )
    model = models.build_model("holt_winters").fit(y)
    point = model.predict(horizon)
    cycle_sigma = stdev(errors) if len(errors) > 1 else 0.0
    plans = [
        models.recommend_cash_load(
            point, model._sigma, service_level, cycle_sigma=cycle_sigma
        )
        for service_level in SERVICE_LEVELS
    ]
    return {
        "observations": len(y),
        "folds": len(errors),
        "cycle_sigma": cycle_sigma,
        "forecast_total": sum(point),
        "plans": plans,
    }


def render_report(results, horizon, min_train, step, excluded):
    lines = [
        "# Real-data cash-loading policy",
        "",
        f"The point forecast uses Holt-Winters. Cycle-error spread uses {horizon}-day rolling-origin forecasts with a {min_train}-day minimum training window and {step}-day origin step.",
        "The source unit is retained; no currency conversion is applied.",
        "",
        f"ATM IDs excluded before policy calculation: {', '.join(excluded) if excluded else 'none'}.",
        "",
        "| ATM | Folds | Forecast total | Cycle-error sigma | Service level | Safety stock | Recommended cycle load |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for atm_id, result in results.items():
        for plan in result["plans"]:
            lines.append(
                f"| {atm_id} | {result['folds']} | {result['forecast_total']:.2f} | "
                f"{result['cycle_sigma']:.2f} | {plan['service_level']:.1%} | "
                f"{plan['safety_stock']:.2f} | {plan['cycle_load']:.2f} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Higher service levels increase the recommended cycle load because the policy adds more safety stock.",
        "- The cycle-error sigma is measured from total 14-day errors, not inferred from independent daily errors.",
        "- This report estimates a target load only. It does not estimate a top-up because the public data has no current balance or replenishment schedule.",
        "- ATM3 was excluded because the EDA found 362 zero days out of 365. ATM4's large observation was retained.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate real-data cash policy targets")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--min-train", type=int, default=180)
    parser.add_argument("--step", type=int, default=14)
    args = parser.parse_args()
    results = {}
    for atm_id in data.list_atms(args.input):
        if atm_id not in args.exclude:
            _, values = data.load_series(args.input, atm_id)
            results[atm_id] = plan_for_atm(values, args.horizon, args.min_train, args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_report(results, args.horizon, args.min_train, args.step, args.exclude),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
