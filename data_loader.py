"""Small standard-library loader for public or private ATM CSV files."""
from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

DATE_FORMATS = (
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def parse_date(value: str) -> date:
    """Parse the date formats used by the bundled and public ATM files."""
    value = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError(f"unsupported date: {value!r}") from exc


def parse_amount(value: str) -> float:
    value = value.strip().replace(",", "")
    if not value:
        raise ValueError("empty amount")
    return float(value)


def load_transactions(
    path: str | Path,
    date_column: str = "DATE",
    atm_column: str = "ATM",
    amount_column: str = "Cash",
):
    """Load usable rows and a small data-quality report from a CSV file.

    Rows with missing or invalid date, ATM, or amount values are skipped. Exact
    duplicate source rows are skipped before the daily aggregation step.
    """
    report = {
        "input_rows": 0,
        "duplicate_rows": 0,
        "skipped_rows": 0,
        "missing_date": 0,
        "missing_atm": 0,
        "missing_amount": 0,
        "invalid_date": 0,
        "invalid_amount": 0,
    }
    rows = []
    seen = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {date_column, atm_column, amount_column}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        for source in reader:
            report["input_rows"] += 1
            key = tuple(source.get(name, "") for name in reader.fieldnames or ())
            if key in seen:
                report["duplicate_rows"] += 1
                continue
            seen.add(key)

            raw_date = (source.get(date_column) or "").strip()
            atm_id = (source.get(atm_column) or "").strip()
            raw_amount = (source.get(amount_column) or "").strip()
            if not raw_date:
                report["missing_date"] += 1
            if not atm_id:
                report["missing_atm"] += 1
            if not raw_amount:
                report["missing_amount"] += 1
            if not raw_date or not atm_id or not raw_amount:
                report["skipped_rows"] += 1
                continue
            try:
                parsed_date = parse_date(raw_date)
            except ValueError:
                report["invalid_date"] += 1
                report["skipped_rows"] += 1
                continue
            try:
                amount = parse_amount(raw_amount)
            except ValueError:
                report["invalid_amount"] += 1
                report["skipped_rows"] += 1
                continue
            rows.append({"date": parsed_date, "atm_id": atm_id, "amount": amount})
    report["usable_rows"] = len(rows)
    return rows, report
