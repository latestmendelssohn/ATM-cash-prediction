# Predicting ATM Cash Balance using Time Series

**A project that forecasts daily ATM cash demand, quantify its
uncertainty, and turn that into a cost-optimal cash-loading decision, with a
Gemini-powered RAG analyst.**

Every ATM loses money two ways: **stock runs outs** (the machine runs dry — lost
customers) and **idle cash** (over-loaded machines tie up
currency that earns nothing). Both come from not knowing tomorrow's demand.
This project builds a time-series forecaster for daily withdrawals,
attaches a 95% prediction interval, and converts it into a replenishment policy
using inventory theory.


| file | contents |
|---|---|
| `data.py` | synthetic ATM data generator + loaders (pure stdlib) |
| `models.py` | metrics, baselines, **Holt-Winters (from scratch)**, SARIMA, backtesting, cash-plan optimiser |
| `analyst.py` | RAG layer: report building + ChromaDB + Gemini + LangChain |
| `app.py` | FastAPI streaming API **and** the command-line interface |
| `test_core.py` | tests for the core math |

## Problem

For ATM $i$ on day $t$, let $y_t$ be the net cash dispensed. Assume we want a 14-day
point forecast $\hat y_{t+1..t+H}$, 95% prediction interval, and cash $S$
to load so that the stock-out probability over the cycle stays below $1-\text{SL}$.
ATM demand is dominated by **calendar structure**: a strong weekend peaks, a monthly salary/rent cycle, and festival spikes.

## My Methods

- **Baselines** — historical mean and **seasonal naïve** ($\hat y_{t+k}=y_{t+k-m}$, $m=7$). Any real model must beat these.
- **Holt-Winters** (implemented from scratch) — triple exponential smoothing over level $\ell_t$, trend $b_t$ and season $s_t$:

  $$\ell_t=\alpha(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1}),\quad
    b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)b_{t-1},$$
  $$s_t=\gamma(y_t-\ell_t)+(1-\gamma)s_{t-m},\qquad
    \hat y_{t+h}=\ell_t+h\,b_t+s_{t-m+((h-1)\bmod m)+1}.$$

  The parameters $(\alpha,\beta,\gamma)$ are fitted by grid search + coordinate descent on the in-sample one-step SSE.
- **SARIMA** — seasonal ARIMA $(p,d,q)(P,D,Q)_m$ via statsmodels (optional).
- **Evaluation** — **rolling-origin backtesting**: slide the forecast origin forward, refit on the past, score the next $H$ days. Metrics: MAE, RMSE, MAPE, sMAPE and **MASE** (MASE < 1 ⇒ beats seasonal naïve).

## From forecast to decision

Given per-rupee stock-out cost $C_u$ and holding cost $C_o$, the cost-optimal
service level is the **newsvendor ratio** $p^\*=C_u/(C_u+C_o)$, and the cash to
load for an $L$-day cycle is the base-stock level

$$S=\sum_{k=1}^{L}\hat y_{t+k} + \underbrace{z_{p^\*}\,\hat\sigma\sqrt{L}}_{\text{safety stock}},$$

with $z_{p^\*}$ the standard-normal quantile (computed from scratch).

## Results (reproducible)

52-fold rolling-origin backtest, $H=14$, on the bundled dataset — **Holt-Winters
wins every ATM**:

| ATM | best model | MASE |
|---|---|---:|
| ATM001 | holt_winters | 0.976 |
| ATM002 | holt_winters | 0.977 |
| ATM003 | holt_winters | 0.865 |
| ATM004 | holt_winters | 0.994 |
| ATM005 | holt_winters | 0.921 |

ATM001 leaderboard: holt_winters MASE **0.976** < seasonal_naive 1.083 < mean 1.572.
Reproduce with `python app.py backtest --atm ATM001` and `python app.py pipeline`.

## Quickstart

```bash
pip install -r requirements.txt          # core runs even without this

python data.py                           # (re)generate the dataset
python app.py forecast  --atm ATM001     # 14-day forecast + interval
python app.py backtest  --atm ATM001     # model leaderboard
python app.py cash-plan --atm ATM001 --balance 1.5e7
python app.py pipeline                   # every ATM at once
pytest                                   # run the core tests
```

### RAG analyst (needs a Gemini key)

```bash
cp .env.example .env        # add GOOGLE_API_KEY
python app.py index --atm ATM001
python app.py chat "How much cash should I load into ATM001 next week?"
```

### API

```bash
uvicorn app:app --reload    # or: python app.py serve   /   docker build -t atmf . && docker run -p 8000:8000 atmf
```

Endpoints: `POST /forecast`, `POST /cash-plan`, `POST /index/{atm_id}`,
`POST /ingest/pdf`, `POST /chat`, `POST /chat/stream` (SSE). Docs at `/docs`.

## Notes

The data generator, metrics, baselines, Holt-Winters, backtester and cash
optimiser are **pure standard-library Python** (no numpy/pandas) and covered by
`test_core.py`. SARIMA, the RAG analyst and the API build on the packages in
`requirements.txt`. The synthetic dataset stands in for confidential ATM logs —
swap in real data by matching the CSV schema (`date, atm_id, net_cash_out, …`).

*References: Hyndman & Athanasopoulos, Forecasting: Principles and Practice;
Box & Jenkins (SARIMA); Hyndman & Koehler 2006 (MASE). Method descriptions
paraphrased for compliance.*
