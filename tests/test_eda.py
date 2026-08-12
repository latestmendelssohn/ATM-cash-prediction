import csv

import eda


def write_processed(path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["date", "atm_id", "net_cash_out"])
        writer.writeheader()
        writer.writerows([
            {"date": "2024-01-01", "atm_id": "A", "net_cash_out": "10"},
            {"date": "2024-01-02", "atm_id": "A", "net_cash_out": "20"},
            {"date": "2024-01-04", "atm_id": "A", "net_cash_out": "40"},
            {"date": "2024-01-01", "atm_id": "B", "net_cash_out": "5"},
        ])


def test_summary_reports_per_atm_calendar_gap(tmp_path):
    path = tmp_path / "daily.csv"
    write_processed(path)
    summary = eda.summarize(eda.read_daily(path))
    assert summary["rows"] == 4
    assert summary["summary"]["A"]["missing_days"] == 1
    assert summary["summary"]["B"]["rows"] == 1


def test_report_contains_data_quality_and_weekday_sections(tmp_path):
    path = tmp_path / "daily.csv"
    write_processed(path)
    report = eda.render_report(eda.summarize(eda.read_daily(path)))
    assert "## Per-ATM summary" in report
    assert "## Mean withdrawal by weekday" in report
    assert "ATM1" not in report
