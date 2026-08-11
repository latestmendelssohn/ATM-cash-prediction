"""
Common forecasting interface + a tiny result container.

Every model in this package follows the same lifecycle::

    model = SomeModel(**hyperparams)
    model.fit(y_train)                # y_train : list[float]
    yhat = model.predict(h)           # -> list[float] of length h

Optionally a model may expose ``predict_interval(h, level)`` returning
``(point, lower, upper)`` for probabilistic forecasts. A default residual-based
Gaussian interval is provided in ``interval_from_residuals`` so even the simple
baselines can produce prediction bands for the cash-inventory optimiser.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class BaseForecaster(Protocol):
    """Structural type shared by all forecasters."""

    def fit(self, y: Sequence[float]) -> "BaseForecaster": ...
    def predict(self, h: int) -> List[float]: ...


@dataclass
class ForecastResult:
    """Container for a fitted model's forecast over a horizon."""

    model: str
    point: List[float]
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None
    level: float = 0.95
    params: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return len(self.point)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Two-sided standard-normal quantiles for common nominal coverage levels.
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.975: 2.2414, 0.99: 2.5758}


def z_for_level(level: float) -> float:
    """Return z_{(1+level)/2}; nearest tabulated value, default 95%."""
    if level in _Z:
        return _Z[level]
    # linear-ish fallback: pick the closest tabulated level
    closest = min(_Z, key=lambda L: abs(L - level))
    return _Z[closest]


def interval_from_residuals(
    point: Sequence[float],
    residual_std: float,
    level: float = 0.95,
    grow_with_horizon: bool = True,
) -> Tuple[List[float], List[float]]:
    r"""Symmetric Gaussian prediction interval around the point forecast.

    For a horizon step ``k`` (1-indexed) the half-width is

    .. math:: z_{(1+\text{level})/2}\;\hat\sigma\;\sqrt{k}

    when ``grow_with_horizon`` is True (random-walk-like error accumulation),
    otherwise a constant ``z * sigma`` band.
    """
    z = z_for_level(level)
    lower, upper = [], []
    for k, p in enumerate(point, start=1):
        width = z * residual_std * (math.sqrt(k) if grow_with_horizon else 1.0)
        lower.append(p - width)
        upper.append(p + width)
    return lower, upper


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: Sequence[float], ddof: int = 1) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - ddof))
