r"""
Holt-Winters triple exponential smoothing -- implemented from scratch
====================================================================

Given a series :math:`y_1,\dots,y_n` with season length :math:`m`, the model
maintains three latent states: **level** :math:`\ell_t`, **trend** :math:`b_t`
and **seasonal** :math:`s_t`.

Additive seasonality
--------------------
.. math::
    \ell_t &= \alpha\,(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1}+b_{t-1})\\
    b_t     &= \beta\,(\ell_t-\ell_{t-1}) + (1-\beta)\,b_{t-1}\\
    s_t     &= \gamma\,(y_t-\ell_t) + (1-\gamma)\,s_{t-m}\\
    \hat y_{t+h} &= \ell_t + h\,b_t + s_{t-m+((h-1)\bmod m)+1}

Multiplicative seasonality
--------------------------
.. math::
    \ell_t &= \alpha\,\frac{y_t}{s_{t-m}} + (1-\alpha)(\ell_{t-1}+b_{t-1})\\
    s_t     &= \gamma\,\frac{y_t}{\ell_t} + (1-\gamma)\,s_{t-m}\\
    \hat y_{t+h} &= (\ell_t + h\,b_t)\;s_{t-m+((h-1)\bmod m)+1}

The smoothing parameters :math:`(\alpha,\beta,\gamma)\in[0,1]^3` are chosen to
minimise the in-sample one-step SSE.  Because SciPy is intentionally *not* a
dependency of the core, optimisation uses a coarse grid followed by a few
rounds of coordinate descent -- enough to land near the optimum for this
smooth, low-dimensional objective.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .base import interval_from_residuals, std


class HoltWinters:
    def __init__(
        self,
        season_length: int = 7,
        trend: str = "add",          # "add" | None
        seasonal: str = "add",       # "add" | "mul" | None
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        optimize: bool = True,
    ) -> None:
        if seasonal not in ("add", "mul", None):
            raise ValueError("seasonal must be 'add', 'mul' or None")
        if trend not in ("add", None):
            raise ValueError("trend must be 'add' or None")
        self.m = season_length
        self.trend = trend
        self.seasonal = seasonal
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.optimize = optimize
        # fitted state
        self._level = 0.0
        self._trend = 0.0
        self._season: List[float] = []
        self._resid_std = 0.0
        self._fitted: List[float] = []

    # ------------------------------------------------------------------ init
    def _initial_states(self, y: Sequence[float]) -> Tuple[float, float, List[float]]:
        m = self.m
        if self.seasonal is None:
            level0 = y[0]
            trend0 = (y[1] - y[0]) if (self.trend and len(y) > 1) else 0.0
            return level0, trend0, [0.0] * m if self.seasonal == "add" else [1.0] * m

        # level: mean of first season
        season1 = y[:m]
        level0 = sum(season1) / m
        # trend: average per-step change between first two seasons
        if self.trend and len(y) >= 2 * m:
            season2 = y[m : 2 * m]
            trend0 = (sum(season2) / m - sum(season1) / m) / m
        else:
            trend0 = 0.0
        # seasonal indices from the first season relative to its mean
        if self.seasonal == "add":
            season0 = [y[i] - level0 for i in range(m)]
        else:  # multiplicative
            season0 = [y[i] / level0 if level0 != 0 else 1.0 for i in range(m)]
        return level0, trend0, season0

    # --------------------------------------------------------------- recursion
    def _run(
        self, y: Sequence[float], alpha: float, beta: float, gamma: float
    ) -> Tuple[float, float, List[float], List[float]]:
        """Run the filter; return final (level, trend, season buffer, fitted)."""
        m = self.m
        level, trend, season = self._initial_states(y)
        season = list(season)
        fitted: List[float] = []

        use_trend = self.trend is not None
        mul = self.seasonal == "mul"
        has_season = self.seasonal is not None

        for t in range(len(y)):
            s_idx = t % m
            s_prev = season[s_idx] if has_season else (1.0 if mul else 0.0)

            # one-step-ahead forecast for time t (before seeing y[t])
            if has_season:
                yhat = (level + (trend if use_trend else 0.0)) * s_prev if mul else \
                       (level + (trend if use_trend else 0.0)) + s_prev
            else:
                yhat = level + (trend if use_trend else 0.0)
            fitted.append(yhat)

            # update states with observed y[t]
            prev_level = level
            if has_season:
                if mul:
                    level = alpha * (y[t] / s_prev if s_prev != 0 else y[t]) + (1 - alpha) * (
                        prev_level + (trend if use_trend else 0.0)
                    )
                else:
                    level = alpha * (y[t] - s_prev) + (1 - alpha) * (
                        prev_level + (trend if use_trend else 0.0)
                    )
            else:
                level = alpha * y[t] + (1 - alpha) * (prev_level + (trend if use_trend else 0.0))

            if use_trend:
                trend = beta * (level - prev_level) + (1 - beta) * trend

            if has_season:
                if mul:
                    season[s_idx] = gamma * (y[t] / level if level != 0 else 1.0) + (1 - gamma) * s_prev
                else:
                    season[s_idx] = gamma * (y[t] - level) + (1 - gamma) * s_prev

        return level, trend, season, fitted

    # ----------------------------------------------------------- optimisation
    def _sse(self, y: Sequence[float], params: Tuple[float, float, float]) -> float:
        a, b, g = params
        _, _, _, fitted = self._run(y, a, b, g)
        # ignore the first season (warm-up) when scoring
        start = self.m if self.seasonal is not None else 1
        return sum((y[t] - fitted[t]) ** 2 for t in range(start, len(y)))

    def _optimise(self, y: Sequence[float]) -> Tuple[float, float, float]:
        grid = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
        beta_grid = grid if self.trend else [0.0]
        gamma_grid = grid if self.seasonal else [0.0]

        best = None
        best_sse = math.inf
        for a in grid:
            for b in beta_grid:
                for g in gamma_grid:
                    sse = self._sse(y, (a, b, g))
                    if sse < best_sse:
                        best_sse, best = sse, (a, b, g)

        # coordinate descent refinement around the grid winner
        a, b, g = best  # type: ignore[misc]
        step = 0.1
        for _ in range(30):
            improved = False
            for i, (lo, active) in enumerate(
                [(True, True), (True, self.trend is not None), (True, self.seasonal is not None)]
            ):
                if not active:
                    continue
                for delta in (step, -step):
                    cand = [a, b, g]
                    cand[i] = min(0.999, max(0.001, cand[i] + delta))
                    sse = self._sse(y, (cand[0], cand[1], cand[2]))
                    if sse < best_sse - 1e-9:
                        best_sse = sse
                        a, b, g = cand
                        improved = True
            if not improved:
                step /= 2
                if step < 1e-3:
                    break
        return a, b, g

    # ------------------------------------------------------------------- API
    def fit(self, y: Sequence[float]) -> "HoltWinters":
        y = list(y)
        if self.seasonal is not None and len(y) < 2 * self.m:
            raise ValueError("need at least two full seasons to fit a seasonal model")
        self._y = y

        if self.optimize and (self.alpha is None or self.beta is None or self.gamma is None):
            a, b, g = self._optimise(y)
        else:
            a = 0.3 if self.alpha is None else self.alpha
            b = 0.1 if self.beta is None else self.beta
            g = 0.1 if self.gamma is None else self.gamma
        self.alpha, self.beta, self.gamma = a, b, g

        level, trend, season, fitted = self._run(y, a, b, g)
        self._level, self._trend, self._season, self._fitted = level, trend, season, fitted

        start = self.m if self.seasonal is not None else 1
        residuals = [y[t] - fitted[t] for t in range(start, len(y))]
        self._resid_std = std(residuals) if len(residuals) > 1 else 0.0
        return self

    def predict(self, h: int) -> List[float]:
        m = self.m
        n = len(self._y)
        use_trend = self.trend is not None
        mul = self.seasonal == "mul"
        has_season = self.seasonal is not None

        out: List[float] = []
        for k in range(1, h + 1):
            base = self._level + (k * self._trend if use_trend else 0.0)
            if has_season:
                s_idx = (n + k - 1) % m
                s = self._season[s_idx]
                out.append(base * s if mul else base + s)
            else:
                out.append(base)
        return out

    def predict_interval(
        self, h: int, level: float = 0.95
    ) -> Tuple[List[float], List[float], List[float]]:
        point = self.predict(h)
        lower, upper = interval_from_residuals(point, self._resid_std, level)
        return point, lower, upper

    @property
    def params(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "season_length": self.m,
            "resid_std": self._resid_std,
        }
