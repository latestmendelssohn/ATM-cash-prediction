import interval_evaluation as E


class FixedModel:
    def fit(self, values):
        return self

    def predict_interval(self, horizon, level):
        point = [10.0] * horizon
        return point, [9.0] * horizon, [11.0] * horizon


def test_interval_metrics_counts_coverage_by_horizon():
    result = E.interval_metrics([10.0] * 20, FixedModel, horizon=3, min_train=5, step=3)
    assert result["folds"] == 5
    assert result["coverage"] == 1.0
    assert result["by_horizon"][0]["mean_width"] == 2.0


def test_report_mentions_nominal_level_and_excluded_atm():
    report = E.render_report(
        {"ATM1": {"folds": 1, "coverage": 0.95, "mean_width": 2.0,
                  "by_horizon": [{"horizon": 1, "coverage": 1.0, "mean_width": 2.0}]}},
        horizon=1, min_train=5, step=1, level=0.95, excluded=["ATM3"],
    )
    assert "Nominal coverage: 95%" in report
    assert "ATM3" in report
