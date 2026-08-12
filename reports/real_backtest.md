# Real-data forecast comparison

This preliminary comparison reuses the project's existing rolling-origin backtester.
The horizon is 14 days, the minimum training window is 180 days, and the origin moves by 14 days.

ATM IDs excluded before modelling: ATM3.
ATM3 is excluded because the EDA found 362 zero days out of 365.

## ATM1

Usable observations: 362

| model | MAE | RMSE | MAPE | sMAPE | MASE | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| holt_winters | 17.70 | 23.34 | 136.75 | 28.92 | 0.91 | 13 |
| seasonal_naive | 19.70 | 25.78 | 123.69 | 28.52 | 1.01 | 13 |
| mean | 25.58 | 35.36 | 232.42 | 38.90 | 1.32 | 13 |

Lower scores are better. MASE below 1 means the model beats the seasonal-naive scale.

## ATM2

Usable observations: 363

| model | MAE | RMSE | MAPE | sMAPE | MASE | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seasonal_naive | 17.85 | 24.42 | 54945159.80 | 46.77 | 0.79 | 13 |
| holt_winters | 18.15 | 23.26 | 479058735.01 | 48.72 | 0.80 | 13 |
| mean | 32.19 | 37.65 | 3483860410.70 | 67.00 | 1.42 | 13 |

Lower scores are better. MASE below 1 means the model beats the seasonal-naive scale.

## ATM4

Usable observations: 365

| model | MAE | RMSE | MAPE | sMAPE | MASE | folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | 335.63 | 519.94 | 553.26 | 74.38 | 0.91 | 13 |
| holt_winters | 362.36 | 584.34 | 708.28 | 73.57 | 0.97 | 13 |
| seasonal_naive | 418.03 | 639.59 | 433.48 | 87.47 | 1.13 | 13 |

Lower scores are better. MASE below 1 means the model beats the seasonal-naive scale.

## Limitations of this first run

- The two missing ATM1 days and three missing ATM2 days were skipped by preprocessing, so the series is slightly shorter than the calendar range.
- ATM4's 10,920 maximum was retained. It should be treated as a data-quality decision, not silently removed after seeing the scores.
- MAPE is not reliable for ATM2 because zero withdrawals make percentage errors explode; MASE, MAE, RMSE, and sMAPE are more useful here.
- The one-year source limits the training window and the number of rolling folds.
