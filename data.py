"""
Synthetic ATM cash data: generation + loading  (pure standard library).
=======================================================================

Real ATM transaction logs are confidential, so we simulate a small fleet with a
*known* data-generating process and then check whether our models recover it.
For ATM ``i`` on day ``t`` the net cash dispensed is

    D[i,t] = base_i * trend(t) * weekday(t) * salary(t) * festival(t) * noise

with a strong weekly cycle (weekend peaks), a monthly salary/rent cycle, Indian
festival spikes and a log-normal shock. The end-of-day balance follows a simple
(s, S) replenishment rule.

Run directly to (re)generate the dataset:

    python data.py --out data/atm_transactions.csv --atms 5 --days 1095
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Day-of-week multipliers (Mon=0 ... Sun=6): strong weekend peak, mid-week trough.
WEEKDAY_MULT = {0: 1.05, 1: 0.82, 2: 0.80, 3: 0.85, 4: 1.20, 5: 1.55, 6: 1.35}

# ATM location archetypes: base daily demand (INR), annual growth, noise sigma.
LOCATIONS = {
    "metro_retail": (850_000, 0.08, 0.06),
    "urban_residential": (520_000, 0.05, 0.07),
    "corporate_park": (640_000, 0.03, 0.09),
    "rural_branch": (280_000, 0.10, 0.10),
    "transit_hub": (910_000, 0.06, 0.08),
}

# Approximate Indian festival / holiday demand multipliers, keyed by (month, day).
FESTIVALS = {
    (1, 1): 1.20, (1, 26): 1.10, (3, 8): 1.15, (4, 14): 1.12, (8, 15): 1.10,
    (8, 30): 1.25, (10, 2): 1.10, (10, 24): 1.55, (11, 12): 1.45,
    (12, 25): 1.30, (12, 31): 1.35,
}

# The bundled dataset, resolved relative to this file so the CLI and the tests
# work from any working directory.
DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "atm_transactions.csv"


def _days_in_month(d: date) -> int:
    nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return (nxt - timedelta(days=1)).day


def salary_mult(d: date) -> float:
    """Salary cycle: spikes on the 1st-3rd and at month-end, small mid-month bumps."""
    dom, dim = d.day, _days_in_month(d)
    if dom <= 3:
        return 1.45 - 0.10 * (dom - 1)
    if dom >= dim - 2:
        return 1.30
    if dom in (7, 15):
        return 1.10
    return 1.0


def festival_mult(d: date) -> float:
    return FESTIVALS.get((d.month, d.day), 1.0)


def generate(out_path: Path, n_atms: int = 5, start: date = date(2021, 1, 1),
             n_days: int = 3 * 365, seed: int = 42) -> Dict[str, object]:
    """Generate the panel dataset and write it to ``out_path`` as CSV."""
    rng = random.Random(seed)
    loc_names = list(LOCATIONS)
    rows: List[dict] = []
    total_stockouts = 0

    for k in range(n_atms):
        loc = loc_names[k % len(loc_names)]
        base0, growth, sigma = LOCATIONS[loc]
        base = base0 * rng.uniform(0.85, 1.15)
        capacity = round(base * 4.0, -4)
        reorder = round(capacity * 0.35, -4)
        atm_id = f"ATM{k + 1:03d}"
        balance = capacity

        for t in range(n_days):
            d = start + timedelta(days=t)
            trend = 1.0 + growth * (t / 365.0)
            demand = (base * trend * WEEKDAY_MULT[d.weekday()] * salary_mult(d)
                      * festival_mult(d) * math.exp(rng.gauss(0.0, sigma)))
            demand = max(0.0, demand)

            replenishment = 0
            if balance < reorder:
                replenishment = int(round(capacity - balance, -3))
                balance += replenishment
            start_balance = balance
            dispensed = min(demand, balance)
            stockout = 1 if demand > balance + 1e-6 else 0
            total_stockouts += stockout
            balance = start_balance - dispensed

            rows.append({
                "date": d.isoformat(),
                "atm_id": atm_id,
                "location_type": loc,
                "net_cash_out": int(round(dispensed, -2)),
                "num_withdrawals": max(1, int(round(dispensed / 3200.0))),
                "is_holiday": 1 if festival_mult(d) > 1.0 else 0,
                "replenishment": replenishment,
                "end_of_day_balance": int(round(balance, -2)),
                "stockout": stockout,
            })

    rows.sort(key=lambda r: (r["date"], r["atm_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {"rows": len(rows), "atms": n_atms, "start": start.isoformat(),
            "end": (start + timedelta(days=n_days - 1)).isoformat(),
            "total_stockouts": total_stockouts, "output": str(out_path)}


def load_series(path: str | Path, atm_id: str,
                target: str = "net_cash_out") -> Tuple[List[date], List[float]]:
    """Read one ATM's univariate daily series, sorted by date (pure stdlib)."""
    with open(path, "r", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["atm_id"] == atm_id]
    if not rows:
        raise ValueError(f"no rows for atm_id={atm_id!r} in {path}")
    rows.sort(key=lambda r: r["date"])
    dates = [datetime.strptime(r["date"], "%Y-%m-%d").date() for r in rows]
    values = [float(r[target]) for r in rows]
    return dates, values


def list_atms(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return sorted({r["atm_id"] for r in csv.DictReader(fh)})


def location_map(path: str | Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["atm_id"], r.get("location_type", ""))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the synthetic ATM dataset.")
    ap.add_argument("--out", default="data/atm_transactions.csv", type=Path)
    ap.add_argument("--atms", type=int, default=5)
    ap.add_argument("--days", type=int, default=3 * 365)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    y, m, d = (int(x) for x in a.start.split("-"))
    summary = generate(a.out, a.atms, date(y, m, d), a.days, a.seed)
    print("Generated synthetic ATM dataset:")
    for key, val in summary.items():
        print(f"  {key:>16}: {val}")
