# Real-data interval coverage

Model: Holt-Winters. Nominal coverage: 95%. Horizon: 14 days. Minimum training window: 180 days. Origin step: 14 days.

ATM IDs excluded before evaluation: ATM3.

| ATM | Folds | Empirical coverage | Mean interval width |
| --- | ---: | ---: | ---: |
| ATM1 | 13 | 99.45% | 270.08 |
| ATM2 | 13 | 98.90% | 297.79 |
| ATM4 | 13 | 99.45% | 5024.79 |

## Coverage by forecast day

### ATM1

| Day | Coverage | Mean width |
| ---: | ---: | ---: |
| 1 | 100.00% | 103.32 |
| 2 | 100.00% | 146.12 |
| 3 | 100.00% | 178.96 |
| 4 | 92.31% | 206.64 |
| 5 | 100.00% | 231.03 |
| 6 | 100.00% | 253.08 |
| 7 | 100.00% | 273.36 |
| 8 | 100.00% | 292.23 |
| 9 | 100.00% | 309.96 |
| 10 | 100.00% | 326.73 |
| 11 | 100.00% | 342.68 |
| 12 | 100.00% | 357.91 |
| 13 | 100.00% | 372.53 |
| 14 | 100.00% | 386.59 |

### ATM2

| Day | Coverage | Mean width |
| ---: | ---: | ---: |
| 1 | 84.62% | 113.92 |
| 2 | 100.00% | 161.11 |
| 3 | 100.00% | 197.32 |
| 4 | 100.00% | 227.84 |
| 5 | 100.00% | 254.74 |
| 6 | 100.00% | 279.05 |
| 7 | 100.00% | 301.41 |
| 8 | 100.00% | 322.22 |
| 9 | 100.00% | 341.77 |
| 10 | 100.00% | 360.25 |
| 11 | 100.00% | 377.84 |
| 12 | 100.00% | 394.64 |
| 13 | 100.00% | 410.75 |
| 14 | 100.00% | 426.26 |

### ATM4

| Day | Coverage | Mean width |
| ---: | ---: | ---: |
| 1 | 100.00% | 1922.25 |
| 2 | 100.00% | 2718.47 |
| 3 | 100.00% | 3329.43 |
| 4 | 100.00% | 3844.50 |
| 5 | 100.00% | 4298.28 |
| 6 | 100.00% | 4708.53 |
| 7 | 92.31% | 5085.79 |
| 8 | 100.00% | 5436.94 |
| 9 | 100.00% | 5766.75 |
| 10 | 100.00% | 6078.68 |
| 11 | 100.00% | 6375.38 |
| 12 | 100.00% | 6658.87 |
| 13 | 100.00% | 6930.77 |
| 14 | 100.00% | 7192.40 |

## Interpretation notes

- Coverage near the nominal level is desirable, but wider intervals are not automatically better.
- ATM4's 10,920 observation is retained. It can widen the residual estimate and affect coverage.
- ATM3 was excluded because the EDA found 362 zero days out of 365.
