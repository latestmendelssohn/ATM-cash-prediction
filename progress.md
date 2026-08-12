# Progress

## 2026-08-12

- Inspected the audit/action plan, repository, tests, and requested skills.
- Baseline: `pytest -q` passed 28 tests.
- Searched for public ATM data and selected the `ATMData.csv` mirror from `Stevee-G/Data624`.
- Verified locally: 1,474 source rows, four ATM IDs, dates from 2009-05-01 to 2010-04-30, 19 missing cash values, and 14 rows with a missing ATM ID.
- Added a standard-library CSV loader and daily preprocessing pipeline. It skips unusable rows, removes exact duplicates, aggregates ATM/day values, reports calendar gaps, and adds past-only calendar, lag, and rolling features.
- Raw data is kept under ignored `data/raw/` because the source repository does not state a clear data licence. Source notes and the output schema are in `data_schema.md`.
- Next: run the real-data preprocessing smoke test, validate the new tests, and commit this step before moving to real-data EDA and model evaluation.


- Preprocessing smoke test passed: 1,474 input rows produced 1,455 daily rows across ATM1-ATM4; the processed file loads through the existing `data.load_series` helper.
- Validation: `py -3.12 -m pytest -q` passed 32 tests; `git diff --check` reported no whitespace errors.


## 2026-08-12 EDA step

- Added `eda.py` and `tests/test_eda.py`. The report stays standard-library only and covers coverage, per-ATM distribution, calendar gaps, weekday means, and month means.
- Generated `reports/real_data_eda.md` from the processed public data.
- Findings: ATM3 has 362 zero days out of 365 and is not a meaningful forecasting series; ATM4 has a maximum of 10,920 against a median of 404 and needs an explicit outlier decision; ATM1 and ATM2 have 3 and 2 internal calendar gaps.
- Validation: `py -3.12 -m pytest -q` passed 34 tests; compilation and `git diff --check` passed.
- Next: run the existing rolling-origin baselines and Holt-Winters on the usable real-data series, keeping ATM3 separate from the main comparison.


## 2026-08-12 real backtest step

- Added `real_backtest.py` and `tests/test_real_backtest.py`. It reuses `models.leaderboard` with a 14-day horizon, 180-day minimum training window, and 14-day origin step.
- Generated `reports/real_backtest.md` for ATM1, ATM2, and ATM4 over 13 folds each. ATM3 was excluded explicitly because of its 362 zero days.
- Results: Holt-Winters has the best ATM1 MASE at 0.91; seasonal naive has the best ATM2 MASE at 0.79; the mean baseline has the best ATM4 MASE at 0.91 while the large ATM4 outlier remains in the data.
- MAPE is documented as unreliable for ATM2 because zero withdrawals make percentage errors explode.
- Validation: `py -3.12 -m pytest -q` passed 36 tests; compilation and `git diff --check` passed.
- Next: decide and document how to handle ATM4's 10,920 outlier before treating the real-data model comparison as final.


## 2026-08-12 interval evaluation step

- Added `interval_evaluation.py` and `tests/test_interval_evaluation.py`. It evaluates empirical coverage and average interval width by forecast day using the same rolling-origin folds.
- Generated `reports/real_interval_coverage.md` for Holt-Winters at nominal 95% coverage, excluding ATM3.
- Measured coverage: ATM1 99.45%, ATM2 98.90%, and ATM4 99.45% over 13 folds each. The intervals are conservative; ATM4's mean interval width is 5,024.79 because the retained outlier inflates uncertainty.
- Validation: `py -3.12 -m pytest -q` passed 38 tests; compilation and `git diff --check` passed.
- Next: use the measured real-data backtest errors in the cash-loading policy, while keeping the stock-out and sparse-series limitations explicit.
