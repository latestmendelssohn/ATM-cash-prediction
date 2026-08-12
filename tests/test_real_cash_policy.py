import real_cash_policy as P


class FixedModel:
    _sigma = 1.0

    def fit(self, values):
        return self

    def predict(self, horizon):
        return [10.0] * horizon


def test_cycle_errors_uses_walk_forward_totals():
    errors = P.cycle_errors([10.0] * 20, FixedModel, horizon=3, min_train=5, step=3)
    assert errors == [0.0] * 5


def test_report_contains_service_levels_and_balance_limit():
    report = P.render_report(
        {"ATM1": {"folds": 2, "forecast_total": 100.0, "cycle_sigma": 5.0,
                  "plans": [{"service_level": 0.95, "safety_stock": 10.0, "cycle_load": 110.0}]}},
        horizon=14, min_train=180, step=14, excluded=["ATM3"],
    )
    assert "95.0%" in report
    assert "does not estimate a top-up" in report
    assert "ATM3" in report
