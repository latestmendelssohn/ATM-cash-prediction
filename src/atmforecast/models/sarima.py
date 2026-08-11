r"""
Seasonal ARIMA(X) forecaster  (statsmodels).
============================================

SARIMA extends ARIMA with a seasonal component. The model is written

    .. math::
        \Phi_P(B^m)\,\phi_p(B)\,(1-B)^d(1-B^m)^D\,y_t
        = \Theta_Q(B^m)\,\theta_q(B)\,\varepsilon_t \;(+\,\beta^\top x_t)

where :math:`B` is the backshift operator, :math:`(p,d,q)` the non-seasonal
orders, :math:`(P,D,Q)_m` the seasonal orders, and :math:`x_t` optional
**exogenous** calendar regressors (SARIMAX). For daily ATM data we use the
weekly season :math:`m=7`; the salary-day / festival dummies enter through
:math:`x_t` so the model can anticipate the deterministic spikes that a pure
ARIMA would smear out.

Order selection: if ``auto=True`` and ``pmdarima`` is installed we let
``auto_arima`` minimise AICc over a search grid; otherwise the fixed orders
from ``config.yaml`` are used. ``check_stationarity`` runs the Augmented
Dickey-Fuller and KPSS tests to justify the differencing order ``d``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .base import interval_from_residuals


def check_stationarity(y: Sequence[float], alpha: float = 0.05) -> dict:
    """Augmented Dickey-Fuller + KPSS tests. Returns p-values and a verdict.

    ADF null  : unit root (non-stationary). Reject (p<alpha) => stationary.
    KPSS null : stationary. Reject (p<alpha) => non-stationary.
    Agreement on 'stationary' is the strongest evidence no differencing is needed.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    adf_stat, adf_p, *_ = adfuller(y, autolag="AIC")
    try:
        kpss_stat, kpss_p, *_ = kpss(y, regression="c", nlags="auto")
    except Exception:  # KPSS can warn/raise on short series
        kpss_stat, kpss_p = float("nan"), float("nan")

    adf_stationary = adf_p < alpha
    kpss_stationary = (kpss_p >= alpha) if kpss_p == kpss_p else None  # nan-safe
    return {
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "kpss_stat": float(kpss_stat),
        "kpss_pvalue": float(kpss_p),
        "adf_says_stationary": bool(adf_stationary),
        "kpss_says_stationary": kpss_stationary,
        "likely_stationary": bool(adf_stationary and (kpss_stationary is not False)),
    }


class SarimaModel:
    """SARIMAX wrapper with the project's standard fit/predict interface."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        auto: bool = False,
        trend: Optional[str] = None,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto = auto
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._result = None
        self._exog_train = None

    def _auto_order(self, y, exog):
        import pmdarima as pm

        m = self.seasonal_order[3]
        model = pm.auto_arima(
            y,
            X=exog,
            seasonal=True,
            m=m,
            information_criterion="aicc",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3, max_q=3, max_P=2, max_Q=2,
        )
        self.order = model.order
        self.seasonal_order = model.seasonal_order
        return model

    def fit(self, y: Sequence[float], exog=None) -> "SarimaModel":
        import numpy as np
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = np.asarray(y, dtype=float)
        self._exog_train = None if exog is None else np.asarray(exog, dtype=float)

        if self.auto:
            try:
                self._auto_order(y, self._exog_train)
            except Exception:
                pass  # fall back to configured orders

        model = SARIMAX(
            y,
            exog=self._exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self._result = model.fit(disp=False)
        return self

    def predict(self, h: int, exog=None) -> List[float]:
        if self._result is None:
            raise RuntimeError("model is not fitted")
        fc = self._result.get_forecast(steps=h, exog=exog)
        return list(map(float, fc.predicted_mean))

    def predict_interval(
        self, h: int, level: float = 0.95, exog=None
    ) -> Tuple[List[float], List[float], List[float]]:
        if self._result is None:
            raise RuntimeError("model is not fitted")
        fc = self._result.get_forecast(steps=h, exog=exog)
        mean = list(map(float, fc.predicted_mean))
        ci = fc.conf_int(alpha=1 - level)
        try:  # numpy array path
            lower = [float(r[0]) for r in ci]
            upper = [float(r[1]) for r in ci]
        except Exception:  # pandas DataFrame path
            lower = [float(v) for v in ci.iloc[:, 0]]
            upper = [float(v) for v in ci.iloc[:, 1]]
        return mean, lower, upper

    @property
    def params(self) -> dict:
        out = {"order": self.order, "seasonal_order": self.seasonal_order}
        if self._result is not None:
            out["aic"] = float(self._result.aic)
            out["bic"] = float(self._result.bic)
        return out

    def summary(self) -> str:
        return "" if self._result is None else str(self._result.summary())
