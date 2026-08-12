"""Turn ATM CSV transactions into daily, model-ready rows."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from data_loader import load_transactions

OUTPUT_FIELDS = (
    "date",
    "atm_id",
    "net_cash_out",
    "transaction_count",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "month",
    "is_month_start",
    "is_month_end",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
)


def _mean(values):
    return sum(values) / len(values) if values else None


def _std(values):
    if len(values) < 2:
        return None
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _last_day_of_month(day: date) -> int:
    next_month = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    return (next_month - timedelta(days=1)).day


def daily_rows(transactions):
    """Aggregate transaction rows by ATM and calendar day."""
    grouped = defaultdict(lambda: [0.0, 0])
    for row in transactions:
        key = (row["atm_id"], row["date"])
        grouped[key][0] += row["amount"]
        grouped[key][1] += 1
    return [
        {
            "date": day,
            "atm_id": atm_id,
            "net_cash_out": total,
            "transaction_count": count,
        }
        for (atm_id, day), (total, count) in sorted(grouped.items(), key=lambda item: item[0])
    ]


def add_features(rows):
    """Add calendar and past-only rolling features to daily rows.

    A date lookup is used instead of list positions, so missing calendar days do
    not silently turn into a false lag. The current day's target is never in a
    lag or rolling feature.
    """
    by_atm = defaultdict(dict)
    for row in rows:
        by_atm[row["atm_id"]][row["date"]] = row["net_cash_out"]

    output = []
    for row in rows:
        day = row["date"]
        values = by_atm[row["atm_id"]]
        previous_week = [
            values.get(day - timedelta(days=offset))
            for offset in range(1, 8)
        ]
        previous_week = [value for value in previous_week if value is not None]
        output.append({
            **row,
            "day_of_week": day.weekday(),
            "is_weekend": int(day.weekday() >= 5),
            "day_of_month": day.day,
            "month": day.month,
            "is_month_start": int(day.day == 1),
            "is_month_end": int(day.day == _last_day_of_month(day)),
            "lag_1": values.get(day - timedelta(days=1)),
            "lag_7": values.get(day - timedelta(days=7)),
            "rolling_mean_7": _mean(previous_week),
            "rolling_std_7": _std(previous_week),
        })
    return output


def _missing_dates(rows):
    by_atm = defaultdict(list)
    for row in rows:
        by_atm[row["atm_id"]].append(row["date"])
    missing = 0
    for dates in by_atm.values():
        known = set(dates)
        day = min(dates)
        end = max(dates)
        while day <= end:
            missing += day not in known
            day += timedelta(days=1)
    return missing


def prepare_dataset(path, date_column="DATE", atm_column="ATM", amount_column="Cash"):
    transactions, report = load_transactions(path, date_column, atm_column, amount_column)
    daily = add_features(daily_rows(transactions))
    report.update({
        "daily_rows": len(daily),
        "atms": sorted({row["atm_id"] for row in daily}),
        "missing_calendar_days_between_observations": _missing_dates(daily),
    })
    if daily:
        report["start_date"] = min(row["date"] for row in daily).isoformat()
        report["end_date"] = max(row["date"] for row in daily).isoformat()
    return daily, report


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: row[field].isoformat() if isinstance(row[field], date) else row[field]
                for field in OUTPUT_FIELDS
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare daily ATM data from CSV")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--date-column", default="DATE")
    parser.add_argument("--atm-column", default="ATM")
    parser.add_argument("--amount-column", default="Cash")
    args = parser.parse_args()
    rows, report = prepare_dataset(
        args.input, args.date_column, args.atm_column, args.amount_column
    )
    write_csv(args.output, rows)
    print(json.dumps(report, indent=2))
