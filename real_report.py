"""Generate all real-data reports with one command."""
from __future__ import annotations

import argparse
from pathlib import Path

import data
import eda
import interval_evaluation
import real_backtest
import real_cash_policy

DEFAULT_INPUT = Path("data/processed/atm_daily.csv")
DEFAULT_OUTPUT = Path("reports")
DEFAULT_EXCLUDED = ("ATM3",)


def report_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "eda": output_dir / "real_data_eda.md",
        "backtest": output_dir / "real_backtest.md",
        "intervals": output_dir / "real_interval_coverage.md",
        "cash_policy": output_dir / "real_cash_policy.md",
    }


def generate_reports(
    input_path=DEFAULT_INPUT,
    output_dir=DEFAULT_OUTPUT,
    excluded=DEFAULT_EXCLUDED,
    horizon=14,
    min_train=180,
    step=14,
    level=0.95,
):
    input_path = Path(input_path)
    paths = report_paths(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    paths["eda"].write_text(
        eda.render_report(eda.summarize(eda.read_daily(input_path))),
        encoding="utf-8",
    )

    atms = real_backtest.selected_atms(input_path, excluded)
    paths["backtest"].write_text(
        real_backtest.render_report(
            real_backtest.run_backtest(input_path, atms, horizon, min_train, step),
            horizon, min_train, step, excluded,
        ),
        encoding="utf-8",
    )
    paths["intervals"].write_text(
        interval_evaluation.render_report(
            interval_evaluation.run_evaluation(
                input_path, atms, horizon, min_train, step, level,
            ),
            horizon, min_train, step, level, excluded,
        ),
        encoding="utf-8",
    )

    policy_results = {}
    for atm_id in atms:
        _, values = data.load_series(input_path, atm_id)
        policy_results[atm_id] = real_cash_policy.plan_for_atm(
            values, horizon, min_train, step,
        )
    paths["cash_policy"].write_text(
        real_cash_policy.render_report(
            policy_results, horizon, min_train, step, excluded,
        ),
        encoding="utf-8",
    )
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate all real-data ATM reports")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude", action="append")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--min-train", type=int, default=180)
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--level", type=float, default=0.95)
    args = parser.parse_args()
    paths = generate_reports(
        args.input,
        args.output,
        args.exclude if args.exclude is not None else DEFAULT_EXCLUDED,
        args.horizon,
        args.min_train,
        args.step,
        args.level,
    )
    for path in paths.values():
        print(f"Wrote {path}")
