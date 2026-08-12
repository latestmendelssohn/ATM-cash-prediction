import real_backtest as B


def test_selected_atms_applies_explicit_exclusions(tmp_path):
    path = tmp_path / "daily.csv"
    path.write_text(
        "date,atm_id,net_cash_out\n"
        "2024-01-01,ATM1,10\n"
        "2024-01-01,ATM3,20\n",
        encoding="utf-8",
    )
    assert B.selected_atms(path, ["ATM3"]) == ["ATM1"]


def test_report_records_backtest_settings_and_limitations():
    report = B.render_report(
        {
            "ATM1": {
                "observations": 10,
                "leaderboard": [{
                    "model": "mean", "MAE": 1.0, "RMSE": 2.0,
                    "MAPE": 3.0, "sMAPE": 4.0, "MASE": 0.5, "folds": 2,
                }],
            }
        },
        horizon=14,
        min_train=180,
        step=14,
        excluded=["ATM3"],
    )
    assert "minimum training window is 180 days" in report
    assert "ATM3" in report
    assert "10,920" in report
