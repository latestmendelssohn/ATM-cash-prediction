r"""
Rolling-origin (walk-forward) backtesting  --  pure standard library.
===================================================================

A single train/test split gives one noisy estimate of accuracy. The honest way
to evaluate a forecaster on time-ordered data is **rolling-origin evaluation**
(a.k.a. time-series cross-validation, Hyndman & Athanasopoulos, *FPP*): we slide
the forecast origin forward through the series and, at each origin, fit on the
past and score the next ``horizon`` steps. Averaging the per-fold errors yields
a far more stable ranking of competing models.

    origin o_1 : train y[0:t1]            score y[t1 : t1+H]
    origin o_2 : train y[0:t1+step]       score y[t1+step : t1+step+H]
    ...

We deliberately never let a model see the future: at origin ``t`` only
``y[:t]`` is used for fitting (``refit=True`` re-estimates parameters each fold;
``refit=False`` reuses the first fit for speed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean as _mean
from statistics import pstdev
from typing import Callable, Dict, List, Sequence

from .metrics import all_metrics

# A factory that returns a *fresh* forecaster instance each call.
ModelFactory = Callable[[], object]


@dataclass
class FoldResult:
    origin: int
    metrics: Dict[str, float]


@dataclass
class BacktestReport:
    model: str
    folds: List[FoldResult] = field(default_factory=list)

    def aggregate(self) -> Dict[str, float]:
        """Mean of each metric across folds."""
        if not self.folds:
            return {}
        keys = self.folds[0].metrics.keys()
        return {k: _mean(f.metrics[k] for f in self.folds) for k in keys}

    def dispersion(self) -> Dict[str, float]:
        """Population std-dev of each metric across folds (stability)."""
        if len(self.folds) < 2:
            return {k: 0.0 for k in (self.folds[0].metrics if self.folds else [])}
        keys = self.folds[0].metrics.keys()
        return {k: pstdev([f.metrics[k] for f in self.folds]) for k in keys}


def rolling_origin_backtest(
    y: Sequence[float],
    model_factory: ModelFactory,
    model_name: str,
    horizon: int = 14,
    min_train_size: int = 120,
    step: int = 7,
    seasonality: int = 7,
    max_folds: int | None = None,
) -> BacktestReport:
    """Walk the origin forward and score ``horizon``-step forecasts each time."""
    y = list(y)
    n = len(y)
    report = BacktestReport(model=model_name)

    origin = min_train_size
    fold_count = 0
    while origin + horizon <= n:
        train = y[:origin]
        actual = y[origin : origin + horizon]

        model = model_factory()
        model.fit(train)                      # type: ignore[attr-defined]
        pred = model.predict(horizon)         # type: ignore[attr-defined]

        m = all_metrics(actual, pred, y_train=train, seasonality=seasonality)
        report.folds.append(FoldResult(origin=origin, metrics=m))

        fold_count += 1
        if max_folds is not None and fold_count >= max_folds:
            break
        origin += step

    return report


def compare_models(
    y: Sequence[float],
    factories: Dict[str, ModelFactory],
    horizon: int = 14,
    min_train_size: int = 120,
    step: int = 7,
    seasonality: int = 7,
    max_folds: int | None = None,
) -> Dict[str, BacktestReport]:
    """Backtest several models on the same folds and return their reports."""
    return {
        name: rolling_origin_backtest(
            y, factory, name, horizon, min_train_size, step, seasonality, max_folds
        )
        for name, factory in factories.items()
    }


def leaderboard(reports: Dict[str, BacktestReport], sort_by: str = "MASE") -> List[dict]:
    """Flatten reports into a ranked table (list of dicts), best model first."""
    rows = []
    for name, rep in reports.items():
        agg = rep.aggregate()
        row = {"model": name, "folds": len(rep.folds)}
        row.update({k: round(v, 4) for k, v in agg.items()})
        rows.append(row)
    rows.sort(key=lambda r: r.get(sort_by, float("inf")))
    return rows


def format_leaderboard(rows: List[dict]) -> str:
    """Render the leaderboard as a fixed-width text table (no dependencies)."""
    if not rows:
        return "(no results)"
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(f"{r[c]}") for r in rows)) for c in cols}
    header = "  ".join(c.rjust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for r in rows:
        lines.append("  ".join(f"{r[c]}".rjust(widths[c]) for c in cols))
    return "\n".join(lines)
