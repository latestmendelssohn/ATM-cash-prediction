import csv

import data_preprocess as P
from data_loader import load_transactions, parse_date


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["DATE", "ATM", "Cash"])
        writer.writeheader()
        writer.writerows(rows)


def test_loader_parses_public_date_format():
    assert parse_date("5/1/2009 12:00:00 AM").isoformat() == "2009-05-01"


def test_loader_skips_missing_values_and_exact_duplicates(tmp_path):
    path = tmp_path / "input.csv"
    write_csv(path, [
        {"DATE": "2024-01-01", "ATM": "A", "Cash": "10"},
        {"DATE": "2024-01-01", "ATM": "A", "Cash": "10"},
        {"DATE": "2024-01-01", "ATM": "", "Cash": "4"},
        {"DATE": "2024-01-02", "ATM": "A", "Cash": ""},
    ])
    rows, report = load_transactions(path)
    assert rows == [{"date": parse_date("2024-01-01"), "atm_id": "A", "amount": 10.0}]
    assert report["duplicate_rows"] == 1
    assert report["skipped_rows"] == 2


def test_daily_aggregation_and_features_do_not_use_current_value():
    rows = [
        {"date": parse_date("2024-01-01"), "atm_id": "A", "amount": 10.0},
        {"date": parse_date("2024-01-02"), "atm_id": "A", "amount": 20.0},
        {"date": parse_date("2024-01-08"), "atm_id": "A", "amount": 80.0},
    ]
    daily = P.add_features(P.daily_rows(rows))
    last = daily[-1]
    assert last["net_cash_out"] == 80.0
    assert last["lag_7"] == 10.0
    assert last["rolling_mean_7"] == 15.0
    assert last["rolling_std_7"] is not None
    assert last["rolling_mean_7"] != last["net_cash_out"]


def test_prepare_dataset_reports_calendar_gaps(tmp_path):
    path = tmp_path / "input.csv"
    write_csv(path, [
        {"DATE": "2024-01-01", "ATM": "A", "Cash": "10"},
        {"DATE": "2024-01-03", "ATM": "A", "Cash": "30"},
    ])
    rows, report = P.prepare_dataset(path)
    assert len(rows) == 2
    assert report["missing_calendar_days_between_observations"] == 1
