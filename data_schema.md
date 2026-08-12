# Real-data input schema

## Source selected for the first real-data pass

The local raw file is `data/raw/ATMData.csv`, downloaded from the public GitHub mirror:

`https://github.com/Stevee-G/Data624/blob/main/Project1/ATMData.csv`

The source file contains one daily row per ATM for the period 2009-05-01 to 2010-04-30. The CSV has four ATM IDs plus rows with a missing ATM ID, and some missing cash values. The accompanying forecasting report describes the file as daily cash withdrawals from four ATMs.

The source repository does not state a clear data licence. For that reason, raw data stays in the local `data/raw/` directory and is ignored by Git. The project commits the loader, preprocessing code, tests, and this source note instead.

The CSV does not state a currency or unit in its header. The preprocessing pipeline therefore keeps the source numeric values unchanged. Do not label the results as rupees or convert them until the source unit is confirmed.

## Required CSV columns

| column | meaning | handling |
|---|---|---|
| `DATE` | transaction or observation date, optionally with a time | parsed to a calendar date |
| `ATM` | ATM identifier | blank values are skipped |
| `Cash` | cash withdrawal amount in the source unit | blank or invalid values are skipped |

The loader also accepts different column names through command-line options.

## Output columns

`data_preprocess.py` writes `data/processed/atm_daily.csv` with these fields:

- `date`, `atm_id`, `net_cash_out`, `transaction_count`
- `day_of_week`, `is_weekend`, `day_of_month`, `month`
- `is_month_start`, `is_month_end`
- `lag_1`, `lag_7`, `rolling_mean_7`, `rolling_std_7`

Exact duplicate source rows are removed before ATM/day aggregation. Calendar and rolling features use only dates before the current row. A missing calendar day is not silently treated as zero demand.

## Reproduce locally

Download the raw file into the ignored directory, then run:

```text
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/Stevee-G/Data624/main/Project1/ATMData.csv', 'data/raw/ATMData.csv')"
python data_preprocess.py --input data/raw/ATMData.csv --output data/processed/atm_daily.csv
```


## ATM4 outlier decision

The raw source contains one ATM4 row with `Cash = 10920` on 2010-02-09. ATM4 has 365 raw rows on 365 unique dates, and this value is not an exact duplicate or a daily aggregation of multiple rows. The processed row also has `transaction_count = 1`.

The value is extreme: ATM4's median is 404, the maximum is 27.03 times the median, and the full-series z-score is 16.05. The local source provides no correction flag, alternate value, or confirmed unit that proves the row is invalid. The value is therefore retained in the primary dataset and model results. Its effect on uncertainty and cash-policy recommendations remains a documented limitation; any future correction should use a separately documented source revision rather than an undocumented winsorization or deletion.
