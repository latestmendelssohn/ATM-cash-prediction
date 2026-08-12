# ATM real-data quality and EDA

This report is generated from the processed daily CSV. Values retain the source unit; the source CSV does not identify a currency.

## Coverage

- Rows: 1455
- ATMs: ATM1, ATM2, ATM3, ATM4
- Date range: 2009-05-01 to 2010-04-30

## Per-ATM summary

| ATM | Rows | Start | End | Missing days | Mean | Median | Std dev | Min | Max | Zero days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ATM1 | 362 | 2009-05-01 | 2010-04-30 | 3 | 83.89 | 91.00 | 36.66 | 1.00 | 180.00 | 0 |
| ATM2 | 363 | 2009-05-01 | 2010-04-30 | 2 | 62.58 | 67.00 | 38.90 | 0.00 | 147.00 | 2 |
| ATM3 | 365 | 2009-05-01 | 2010-04-30 | 0 | 0.72 | 0.00 | 7.94 | 0.00 | 96.00 | 362 |
| ATM4 | 365 | 2009-05-01 | 2010-04-30 | 0 | 474.01 | 404.00 | 650.95 | 2.00 | 10920.00 | 0 |

## Mean withdrawal by weekday

| ATM | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ATM1 | 86.00 | 89.57 | 82.15 | 31.69 | 98.64 | 96.61 | 102.65 |
| ATM2 | 58.73 | 73.25 | 43.75 | 25.53 | 92.02 | 75.98 | 67.15 |
| ATM3 | 0.00 | 0.00 | 1.85 | 1.58 | 1.60 | 0.00 | 0.00 |
| ATM4 | 481.33 | 647.25 | 413.67 | 169.88 | 573.51 | 491.48 | 539.06 |

## Mean withdrawal by month

| ATM | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ATM1 | 86.35 | 90.39 | 78.48 | 76.67 | 80.00 | 89.33 | 84.55 | 100.94 | 94.20 | 72.48 | 76.30 | 78.13 |
| ATM2 | 58.42 | 54.18 | 51.68 | 61.40 | 75.42 | 76.04 | 67.55 | 71.94 | 62.23 | 61.48 | 56.40 | 54.45 |
| ATM3 | 0.00 | 0.00 | 0.00 | 8.77 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ATM4 | 480.77 | 824.36 | 429.48 | 395.17 | 407.10 | 458.13 | 467.00 | 513.03 | 591.00 | 306.84 | 430.23 | 418.26 |

## Initial modelling notes

- ATM-level coverage should be checked before fitting a model.
- Missing calendar days are reported rather than silently converted to zero withdrawal.
- Weekday and month means are descriptive only. They are not model evaluation results.
