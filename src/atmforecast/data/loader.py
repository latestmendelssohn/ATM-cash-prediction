"""
Data loading & aggregation.

Two entry points:

* ``load_series`` -- pure-Python (stdlib csv). Returns aligned ``dates`` and
  ``values`` lists for one ATM + one target column. Used by the pure-Python
  forecasting core so it stays dependency-free.

* ``load_panel`` / ``daily_frame`` -- pandas-based helpers for the library
  models (SARIMA, Prophet, LSTM) and for exploratory analysis.
"""
from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_series(
    path: str | Path,
    atm_id: str,
    target: str = "net_cash_out",
) -> Tuple[List[date], List[float]]:
    """Read a single ATM's univariate daily series (pure stdlib).

    Returns
    -------
    (dates, values) : the date index and the target values, sorted by date.
    """
    dates: List[date] = []
    values: List[float] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if target not in reader.fieldnames:  # type: ignore[arg-type]
            raise KeyError(f"target column {target!r} not in {reader.fieldnames}")
        rows = [r for r in reader if r["atm_id"] == atm_id]
    rows.sort(key=lambda r: r["date"])
    for r in rows:
        dates.append(_parse_date(r["date"]))
        values.append(float(r[target]))
    if not values:
        raise ValueError(f"no rows found for atm_id={atm_id!r} in {path}")
    return dates, values


def list_atms(path: str | Path) -> List[str]:
    """Return the sorted unique ATM ids present in the raw file (stdlib)."""
    seen = set()
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            seen.add(r["atm_id"])
    return sorted(seen)


# ---------------------------------------------------------------------------
# pandas-based helpers (library models / EDA)
# ---------------------------------------------------------------------------


def load_panel(path: str | Path):
    """Load the full transaction panel as a pandas DataFrame with a parsed date."""
    import pandas as pd

    df = pd.read_csv(path, parse_dates=["date"])
    return df


def daily_frame(
    path: str | Path,
    atm_id: Optional[str] = None,
    target: str = "net_cash_out",
):
    """Return a daily-indexed pandas DataFrame for one ATM (or the fleet total).

    The index is a complete ``DatetimeIndex`` at daily frequency (gaps, if any,
    are forward/zero filled) which is what statsmodels/Prophet expect.
    """
    import pandas as pd

    df = load_panel(path)
    if atm_id is not None:
        df = df[df["atm_id"] == atm_id]
        s = df.set_index("date")[target].sort_index()
    else:
        s = df.groupby("date")[target].sum().sort_index()

    s = s.asfreq("D")
    s = s.interpolate(method="time").ffill().bfill()
    out = s.to_frame(name=target)
    out.index.name = "date"
    return out


if __name__ == "__main__":  # tiny smoke test (stdlib only)
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "data/raw/atm_transactions.csv"
    atms = list_atms(p)
    ds, vs = load_series(p, atms[0])
    print(f"ATMs: {atms}")
    print(f"{atms[0]}: {len(vs)} days, {ds[0]} .. {ds[-1]}, mean={sum(vs)/len(vs):,.0f}")
