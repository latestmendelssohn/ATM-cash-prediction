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
