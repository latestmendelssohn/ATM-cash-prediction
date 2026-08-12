# Real-data cash-loading policy

The point forecast uses Holt-Winters. Cycle-error spread uses 14-day rolling-origin forecasts with a 180-day minimum training window and 14-day origin step.
The source unit is retained; no currency conversion is applied.

ATM IDs excluded before policy calculation: ATM3.

| ATM | Folds | Forecast total | Cycle-error sigma | Service level | Safety stock | Recommended cycle load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ATM1 | 13 | 1052.98 | 115.42 | 90.0% | 147.92 | 1200.90 |
| ATM1 | 13 | 1052.98 | 115.42 | 95.0% | 189.85 | 1242.83 |
| ATM1 | 13 | 1052.98 | 115.42 | 97.5% | 226.22 | 1279.20 |
| ATM1 | 13 | 1052.98 | 115.42 | 99.0% | 268.50 | 1321.49 |
| ATM2 | 13 | 915.78 | 78.06 | 90.0% | 100.04 | 1015.82 |
| ATM2 | 13 | 915.78 | 78.06 | 95.0% | 128.40 | 1044.18 |
| ATM2 | 13 | 915.78 | 78.06 | 97.5% | 153.00 | 1068.77 |
| ATM2 | 13 | 915.78 | 78.06 | 99.0% | 181.60 | 1097.37 |
| ATM4 | 13 | 6508.24 | 3207.07 | 90.0% | 4110.02 | 10618.26 |
| ATM4 | 13 | 6508.24 | 3207.07 | 95.0% | 5275.15 | 11783.39 |
| ATM4 | 13 | 6508.24 | 3207.07 | 97.5% | 6285.73 | 12793.97 |
| ATM4 | 13 | 6508.24 | 3207.07 | 99.0% | 7460.75 | 13968.99 |

## Interpretation

- Higher service levels increase the recommended cycle load because the policy adds more safety stock.
- The cycle-error sigma is measured from total 14-day errors, not inferred from independent daily errors.
- This report estimates a target load only. It does not estimate a top-up because the public data has no current balance or replenishment schedule.
- ATM3 was excluded because the EDA found 362 zero days out of 365. ATM4's large observation was retained.
