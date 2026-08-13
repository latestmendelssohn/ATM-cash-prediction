r"""
Forecasting models, evaluation, backtesting and cash planning.
==============================================================

Pure-Python core (no third-party deps):
    * metrics: MAE, RMSE, MAPE, sMAPE, MASE
    * SeasonalNaive, MeanForecast (baselines)
    * HoltWinters  -- triple exponential smoothing, implemented from scratch
    * rolling_origin_backtest / leaderboard
    * recommend_cash_load -- newsvendor / safety-stock cash policy

Library-based (needs statsmodels):
    * SarimaModel -- seasonal ARIMA

Every forecaster follows the same lifecycle: ``fit(y).predict(h)`` and, where
useful, ``predict_interval(h, level) -> (point, lower, upper)``.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. Accuracy metrics
# ---------------------------------------------------------------------------

def mae(y, f):  return sum(abs(a - b) for a, b in zip(y, f)) / len(y)
def rmse(y, f): return math.sqrt(sum((a - b) ** 2 for a, b in zip(y, f)) / len(y))


def mape(y, f, eps=1e-8):
    return 100.0 * sum(abs(a - b) / max(abs(a), eps) for a, b in zip(y, f)) / len(y)


def smape(y, f, eps=1e-8):
    return 100.0 * sum(
        0.0 if (abs(a) + abs(b)) < eps else 2 * abs(a - b) / (abs(a) + abs(b))
        for a, b in zip(y, f)
    ) / len(y)


def mase(y, f, y_train, m=7):
    r"""Mean Absolute Scaled Error: MAE scaled by the in-sample seasonal-naive
    error. ``MASE < 1`` means the forecast beats the seasonal-naive benchmark."""
    if len(y_train) <= m:
        raise ValueError("training series too short for seasonality")
    scale = sum(abs(y_train[t] - y_train[t - m]) for t in range(m, len(y_train))) / (len(y_train) - m)
    return float("inf") if scale == 0 else mae(y, f) / scale


def all_metrics(y, f, y_train=None, m=7) -> Dict[str, float]:
    out = {"MAE": mae(y, f), "RMSE": rmse(y, f), "MAPE": mape(y, f), "sMAPE": smape(y, f)}
    if y_train is not None:
        try:
            out["MASE"] = mase(y, f, y_train, m)
        except ValueError:
            out["MASE"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# 2. Prediction-interval helper (Gaussian, widens with horizon)
# ---------------------------------------------------------------------------

def z_for_level(level: float) -> float:
    """Two-sided normal quantile for a confidence ``level`` (0.95 -> 1.96)."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0,1)")
    return normal_quantile(0.5 * (1.0 + level))


def _std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def _interval(point: Sequence[float], sigma: float, level: float):
    z = z_for_level(level)
    lower = [p - z * sigma * math.sqrt(k) for k, p in enumerate(point, 1)]
    upper = [p + z * sigma * math.sqrt(k) for k, p in enumerate(point, 1)]
    return lower, upper


# ---------------------------------------------------------------------------
# 3. Baselines
# ---------------------------------------------------------------------------

class MeanForecast:
    """Forecast every future value as the historical mean."""

    def fit(self, y):
        self._mean = sum(y) / len(y)
        self._sigma = _std(y)
        return self

    def predict(self, h):
        return [self._mean] * h

    def predict_interval(self, h, level=0.95):
        p = self.predict(h)
        lo, hi = _interval(p, self._sigma, level)
        return p, lo, hi


class SeasonalNaive:
    r"""Repeat the last observed season: :math:`\hat y_{n+k}=y_{n+k-m}`."""

    def __init__(self, m=7):
        self.m = m

    def fit(self, y):
        if len(y) < self.m:
            raise ValueError("series shorter than one season")
        self._last = list(y[-self.m:])
        self._sigma = _std([y[t] - y[t - self.m] for t in range(self.m, len(y))])
        return self

    def predict(self, h):
        return [self._last[k % self.m] for k in range(h)]

    def predict_interval(self, h, level=0.95):
        p = self.predict(h)
        lo, hi = _interval(p, self._sigma, level)
        return p, lo, hi


# ---------------------------------------------------------------------------
# 4. Holt-Winters triple exponential smoothing  (from scratch)
# ---------------------------------------------------------------------------

class HoltWinters:
    r"""Additive/multiplicative triple exponential smoothing.

    States: level :math:`\ell_t`, trend :math:`b_t`, season :math:`s_t`.
    Additive updates::

        l_t = a (y_t - s_{t-m}) + (1-a)(l_{t-1}+b_{t-1})
        b_t = B (l_t - l_{t-1}) + (1-B) b_{t-1}
        s_t = g (y_t - l_t)     + (1-g) s_{t-m}
        yhat_{t+h} = l_t + h b_t + s_{t-m+((h-1) mod m)+1}

    Smoothing parameters (a, B, g) are fitted by grid search + coordinate
    descent on the in-sample one-step SSE (no SciPy needed).
    """

    def __init__(self, m=7, trend="add", seasonal="add", optimize=True,
                 alpha=None, beta=None, gamma=None):
        self.m, self.trend, self.seasonal, self.optimize = m, trend, seasonal, optimize
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    # -- initial states --
    def _init(self, y):
        m = self.m
        if self.seasonal is None:
            return y[0], (y[1] - y[0] if self.trend and len(y) > 1 else 0.0), [0.0] * m
        level0 = sum(y[:m]) / m
        trend0 = ((sum(y[m:2 * m]) / m - level0) / m) if (self.trend and len(y) >= 2 * m) else 0.0
        if self.seasonal == "add":
            season0 = [y[i] - level0 for i in range(m)]
        else:
            season0 = [y[i] / level0 if level0 else 1.0 for i in range(m)]
        return level0, trend0, season0

    # -- run the recursion, return final states + one-step fitted values --
    def _run(self, y, a, b, g):
        m, use_t, mul, has_s = self.m, self.trend is not None, self.seasonal == "mul", self.seasonal is not None
        level, trend, season = self._init(y)
        season = list(season)
        fitted = []
        for t in range(len(y)):
            s_prev = season[t % m] if has_s else (1.0 if mul else 0.0)
            base = level + (trend if use_t else 0.0)
            fitted.append(base * s_prev if (has_s and mul) else (base + s_prev if has_s else base))
            prev = level
            if has_s and mul:
                level = a * (y[t] / s_prev if s_prev else y[t]) + (1 - a) * base
            elif has_s:
                level = a * (y[t] - s_prev) + (1 - a) * base
            else:
                level = a * y[t] + (1 - a) * base
            if use_t:
                trend = b * (level - prev) + (1 - b) * trend
            if has_s:
                season[t % m] = (g * (y[t] / level if level else 1.0) + (1 - g) * s_prev) if mul \
                    else (g * (y[t] - level) + (1 - g) * s_prev)
        return level, trend, season, fitted

    def _sse(self, y, p):
        _, _, _, fitted = self._run(y, *p)
        start = self.m if self.seasonal is not None else 1
        return sum((y[t] - fitted[t]) ** 2 for t in range(start, len(y)))

    def _optimise(self, y):
        grid = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
        bg = grid if self.trend else [0.0]
        gg = grid if self.seasonal else [0.0]
        best, best_sse = (0.3, 0.1, 0.1), math.inf
        for a in grid:
            for b in bg:
                for g in gg:
                    sse = self._sse(y, (a, b, g))
                    if sse < best_sse:
                        best_sse, best = sse, (a, b, g)
        a, b, g = best
        step = 0.1
        for _ in range(30):
            improved = False
            for i, active in enumerate((True, self.trend is not None, self.seasonal is not None)):
                if not active:
                    continue
                for delta in (step, -step):
                    cand = [a, b, g]
                    cand[i] = min(0.999, max(0.001, cand[i] + delta))
                    sse = self._sse(y, tuple(cand))
                    if sse < best_sse - 1e-9:
                        best_sse, (a, b, g) = sse, cand
                        improved = True
            if not improved:
                step /= 2
                if step < 1e-3:
                    break
        return a, b, g

    def fit(self, y):
        y = list(y)
        if self.seasonal is not None and len(y) < 2 * self.m:
            raise ValueError("need at least two full seasons")
        self._y = y
        if self.optimize and None in (self.alpha, self.beta, self.gamma):
            self.alpha, self.beta, self.gamma = self._optimise(y)
        else:
            self.alpha = 0.3 if self.alpha is None else self.alpha
            self.beta = 0.1 if self.beta is None else self.beta
            self.gamma = 0.1 if self.gamma is None else self.gamma
        self._level, self._trend, self._season, fitted = self._run(y, self.alpha, self.beta, self.gamma)
        start = self.m if self.seasonal is not None else 1
        self._sigma = _std([y[t] - fitted[t] for t in range(start, len(y))])
        return self

    def predict(self, h):
        m, n = self.m, len(self._y)
        use_t, mul, has_s = self.trend is not None, self.seasonal == "mul", self.seasonal is not None
        out = []
        for k in range(1, h + 1):
            base = self._level + (k * self._trend if use_t else 0.0)
            if has_s:
                s = self._season[(n + k - 1) % m]
                out.append(base * s if mul else base + s)
            else:
                out.append(base)
        return out

    def predict_interval(self, h, level=0.95):
        p = self.predict(h)
        lo, hi = _interval(p, self._sigma, level)
        return p, lo, hi

    @property
    def params(self):
        return {"alpha": round(self.alpha, 3), "beta": round(self.beta, 3),
                "gamma": round(self.gamma, 3), "m": self.m,
                "trend": self.trend, "seasonal": self.seasonal}


# ---------------------------------------------------------------------------
# 5. SARIMA (statsmodels) -- imported lazily so the core stays dependency-free
# ---------------------------------------------------------------------------

class SarimaModel:
    r"""Seasonal ARIMA :math:`(p,d,q)(P,D,Q)_m` via statsmodels SARIMAX."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order, self.seasonal_order, self._res = order, seasonal_order, None

    def fit(self, y):
        import numpy as np
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._res = SARIMAX(np.asarray(y, float), order=self.order,
                            seasonal_order=self.seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
        return self

    def predict(self, h):
        return list(map(float, self._res.get_forecast(steps=h).predicted_mean))

    def predict_interval(self, h, level=0.95):
        fc = self._res.get_forecast(steps=h)
        ci = fc.conf_int(alpha=1 - level)
        lower = [float(r[0]) for r in ci]
        upper = [float(r[1]) for r in ci]
        return list(map(float, fc.predicted_mean)), lower, upper

    @property
    def params(self):
        p = {"order": self.order, "seasonal_order": self.seasonal_order}
        if self._res is not None:
            p["aic"] = round(float(self._res.aic), 1)
        return p


def build_model(name: str, m: int = 7):
    name = name.lower()
    if name == "mean":
        return MeanForecast()
    if name in ("seasonal_naive", "snaive"):
        return SeasonalNaive(m)
    if name in ("holt_winters", "hw"):
        return HoltWinters(m, "add", "add")
    if name == "sarima":
        return SarimaModel(seasonal_order=(1, 1, 1, m))
    raise ValueError(f"unknown model: {name!r}")


# ---------------------------------------------------------------------------
# 6. Rolling-origin backtesting
# ---------------------------------------------------------------------------

def rolling_origin_backtest(y, factory: Callable[[], object], horizon=14,
                            min_train=180, step=14, m=7) -> Dict[str, float]:
    """Walk the forecast origin forward; average per-fold metrics."""
    y = list(y)
    if len(y) < min_train + horizon:
        raise ValueError(f"series too short for backtesting: need at least "
                         f"min_train + horizon = {min_train + horizon} points, got {len(y)}")
    folds: List[Dict[str, float]] = []
    origin = min_train
    while origin + horizon <= len(y):
        model = factory()
        model.fit(y[:origin])
        folds.append(all_metrics(y[origin:origin + horizon], model.predict(horizon),
                                 y_train=y[:origin], m=m))
        origin += step
    keys = folds[0].keys()
    return {k: sum(f[k] for f in folds) / len(folds) for k in keys} | {"folds": len(folds)}


def cycle_sigma_from_backtest(y, factory: Callable[[], object], horizon=14,
                              min_train=180, step=14) -> float:
    r"""Standard deviation of the *cycle-total* forecast error, measured not assumed.

    The per-day interval widens like :math:`\sigma\sqrt{k}`, which assumes errors
    accumulate. The safety stock needs the spread of the total over the whole
    cycle, so measure it directly on the same rolling-origin folds: for each
    fold, compare the actual ``horizon``-day total against the forecast total.
    """
    y = list(y)
    if len(y) < min_train + horizon:
        raise ValueError(f"series too short for backtesting: need at least "
                         f"min_train + horizon = {min_train + horizon} points, got {len(y)}")
    errors: List[float] = []
    origin = min_train
    while origin + horizon <= len(y):
        model = factory()
        model.fit(y[:origin])
        errors.append(sum(y[origin:origin + horizon]) - sum(model.predict(horizon)))
        origin += step
    return _std(errors)


def leaderboard(y, models=("holt_winters", "seasonal_naive", "mean"),
                horizon=14, min_train=180, step=14, m=7) -> List[dict]:
    """Backtest several models on the same folds; return a table sorted by MASE."""
    rows = []
    for name in models:
        agg = rolling_origin_backtest(y, lambda n=name: build_model(n, m),
                                      horizon, min_train, step, m)
        rows.append({"model": name, **{k: round(v, 4) for k, v in agg.items()}})
    rows.sort(key=lambda r: r.get("MASE", float("inf")))
    return rows


def format_table(rows: List[dict]) -> str:
    if not rows:
        return "(no results)"
    cols = list(rows[0])
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = lambda vals: "  ".join(str(v).rjust(w[c]) for c, v in zip(cols, vals))
    return "\n".join([line(cols), "  ".join("-" * w[c] for c in cols)]
                     + [line([r[c] for c in cols]) for r in rows])


# ---------------------------------------------------------------------------
# 7. Cash-inventory optimisation (newsvendor / safety stock)
# ---------------------------------------------------------------------------

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    r"""Inverse standard-normal CDF :math:`z_p` (Acklam's approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def recommend_cash_load(forecast: Sequence[float], sigma: float,
                        service_level: float = 0.95,
                        current_balance: float | None = None,
                        cu: float | None = None, co: float | None = None,
                        cycle_sigma: float | None = None) -> Dict[str, object]:
    r"""Base-stock cash to load for a cycle of length ``L = len(forecast)``::

        S = sum(forecast) + z_{p*} * cycle_sigma

    where the second term is the safety stock buffering demand uncertainty.

    ``cu`` and ``co`` are the per-rupee understock (stock-out) and overstock
    (idle cash) costs. Supply both and the service level becomes the newsvendor
    critical ratio :math:`p^* = C_u / (C_u + C_o)` instead of a number chosen by
    hand. ``cycle_sigma`` is the standard deviation of *cycle-total* demand; pass
    the value measured by :func:`cycle_sigma_from_backtest` when you have it, and
    it falls back to the independent-errors approximation ``sigma * sqrt(L)``.
    """
    if cu is not None and co is not None:
        if cu <= 0 or co <= 0:
            raise ValueError("cu and co must be positive")
        service_level = cu / (cu + co)
    z = normal_quantile(service_level)
    L = len(forecast)
    cycle_mean = sum(forecast)
    if cycle_sigma is None:
        cycle_sigma = sigma * math.sqrt(L)
    cycle_load = cycle_mean + z * cycle_sigma
    stockout_prob = 1.0 - normal_cdf(z) if cycle_sigma else 0.0
    plan = {
        "service_level": service_level,
        "cycle_load": cycle_load,
        "safety_stock": z * cycle_sigma,
        "cycle_sigma": cycle_sigma,
        "per_day_load": [mu + z * sigma for mu in forecast],
        "expected_stockout_prob": stockout_prob,
    }
    if current_balance is not None:
        plan["current_balance"] = current_balance
        plan["suggested_topup"] = max(0.0, cycle_load - current_balance)
    return plan
