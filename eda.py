"""Generate a small Markdown quality and exploratory report for daily ATM data."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from statistics import mean, median, stdev

from data_loader import parse_date


def read_daily(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"date", "atm_id", "net_cash_out"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            rows.append({
                "date": parse_date(row["date"]),
                "atm_id": row["atm_id"],
                "net_cash_out": float(row["net_cash_out"]),
            })
    return rows


def _missing_days(rows):
    dates = {row["date"] for row in rows}
    if not dates:
        return 0
    day, end = min(dates), max(dates)
    missing = 0
    while day <= end:
        missing += day not in dates
        day += timedelta(days=1)
    return missing


def summarize(rows):
    by_atm = defaultdict(list)
    for row in rows:
        by_atm[row["atm_id"]].append(row)

    atm_summary = {}
    weekday_means = {}
    month_means = {}
    for atm_id, atm_rows in sorted(by_atm.items()):
        values = [row["net_cash_out"] for row in atm_rows]
        atm_summary[atm_id] = {
            "rows": len(atm_rows),
            "start": min(row["date"] for row in atm_rows).isoformat(),
            "end": max(row["date"] for row in atm_rows).isoformat(),
            "missing_days": _missing_days(atm_rows),
            "mean": mean(values),
            "median": median(values),
            "stdev": stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
            "zero_days": sum(value == 0 for value in values),
        }
        weekday_values = defaultdict(list)
        month_values = defaultdict(list)
        for row in atm_rows:
            weekday_values[row["date"].strftime("%a")].append(row["net_cash_out"])
            month_values[row["date"].month].append(row["net_cash_out"])
        weekday_means[atm_id] = {
            day: mean(values) for day, values in sorted(weekday_values.items())
        }
        month_means[atm_id] = {
            month: mean(values) for month, values in sorted(month_values.items())
        }

    return {
        "rows": len(rows),
        "atms": sorted(by_atm),
        "start": min(row["date"] for row in rows).isoformat() if rows else None,
        "end": max(row["date"] for row in rows).isoformat() if rows else None,
        "summary": atm_summary,
        "weekday_means": weekday_means,
        "month_means": month_means,
    }


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(summary):
    lines = [
        "# ATM real-data quality and EDA",
        "",
        "This report is generated from the processed daily CSV. Values retain the source unit; the source CSV does not identify a currency.",
        "",
        "## Coverage",
        "",
        f"- Rows: {summary['rows']}",
        f"- ATMs: {', '.join(summary['atms'])}",
        f"- Date range: {summary['start']} to {summary['end']}",
        "",
        "## Per-ATM summary",
        "",
    ]
    headers = ["ATM", "Rows", "Start", "End", "Missing days", "Mean", "Median", "Std dev", "Min", "Max", "Zero days"]
    rows = []
    for atm_id, item in summary["summary"].items():
        rows.append([
            atm_id, item["rows"], item["start"], item["end"], item["missing_days"],
            f"{item['mean']:.2f}", f"{item['median']:.2f}", f"{item['stdev']:.2f}",
            f"{item['minimum']:.2f}", f"{item['maximum']:.2f}", item["zero_days"],
        ])
    lines.append(_table(headers, rows))
    lines.extend(["", "## Mean withdrawal by weekday", ""])
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    lines.append(_table(
        ["ATM"] + weekdays,
        [
            [atm_id] + [f"{values.get(day, 0):.2f}" for day in weekdays]
            for atm_id, values in summary["weekday_means"].items()
        ],
    ))
    lines.extend(["", "## Mean withdrawal by month", ""])
    lines.append(_table(
        ["ATM"] + [str(month) for month in range(1, 13)],
        [
            [atm_id] + [f"{values.get(month, 0):.2f}" for month in range(1, 13)]
            for atm_id, values in summary["month_means"].items()
        ],
    ))
    lines.extend([
        "",
        "## Initial modelling notes",
        "",
        "- ATM-level coverage should be checked before fitting a model.",
        "- Missing calendar days are reported rather than silently converted to zero withdrawal.",
        "- Weekday and month means are descriptive only. They are not model evaluation results.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write an ATM EDA Markdown report")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(summarize(read_daily(args.input))), encoding="utf-8")
    print(f"Wrote {args.output}")
