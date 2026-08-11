# Predicting ATM Cash Balance using Time Series

**An M.Sc. (Mathematics) level project — forecasting daily ATM cash demand and driving cash-replenishment decisions, with a Gemini-powered RAG analyst.**

Banks lose money in two opposite ways at every ATM: **stock-outs** (a machine
runs dry, customers are turned away, the bank incurs SLA penalties and
reputational damage) and **idle cash** (over-loaded machines tie up currency
that earns nothing and costs money to insure and transport). Both failures come
from the same root cause — not knowing tomorrow's demand. This project builds a
statistically principled **time-series forecasting** pipeline for daily ATM cash
withdrawals, quantifies the forecast uncertainty, and converts it into a
**cost-optimal cash-loading policy** using inventory theory. A
Retrieval-Augmented-Generation (RAG) layer then lets an operations manager
interrogate the results in plain English.

> This project intentionally mirrors the architecture of the *FinApp RAG Agent*
> (LangChain + Gemini + ChromaDB + Pandas + PyPDF, Dockerised, session-based,
> streaming API) and applies that same stack to a quantitative forecasting
> problem.

---

## 1. Problem statement

Let $y_t^{(i)}$ be the net cash dispensed by ATM $i$ on day $t$ (in INR). We want,
for a horizon of $H = 14$ days:

1. a **point forecast** $\hat y_{t+1},\dots,\hat y_{t+H}$;
2. a **prediction interval** $[\hat\ell_{t+k}, \hat u_{t+k}]$ at nominal
   coverage $1-\alpha = 95\%$; and
3. a **replenishment decision**: how much cash $S$ to load so that the
   probability of a stock-out over the cycle is at most $1 - \text{SL}$ for a
   target service level SL, at minimum expected cost.

ATM demand is dominated by **calendar effects** rather than by its own recent
history: a strong weekly cycle (weekend peaks, mid-week troughs), a monthly
salary/rent cycle (spikes on the 1st–3rd and at month-end), and festival
spikes (Diwali, Christmas, …). A good model must represent this structure.

## 2. Data

Because real ATM transaction logs are confidential, the project ships a
**reproducible synthetic generator** (`atmforecast.data.generate_synthetic`,
pure standard library) whose data-generating process is known, so we can check
whether each model *recovers* the structure we planted:

$$
D_{i,t} = L_i \cdot \text{Trend}_i(t)\cdot \text{Week}(t)\cdot \text{Month}(t)\cdot \text{Holiday}(t)\cdot \varepsilon_{i,t},\qquad \varepsilon_{i,t}=e^{\mathcal N(0,\sigma_i^2)} .
$$

The end-of-day **cash balance** is then simulated under an $(s, S)$
threshold-replenishment rule, producing the target we ultimately manage. The
bundled dataset (`data/raw/atm_transactions.csv`) covers **5 heterogeneous ATMs
× 1095 days (2021–2023)**.

| column | meaning |
|---|---|
| `net_cash_out` | daily net cash dispensed (INR) — the forecasting target |
| `end_of_day_balance` | cash remaining in the cassette (INR) |
| `replenishment` | cash loaded that day |
| `is_holiday`, `stockout` | event flags |

Regenerate (optionally larger) with:

```bash
python -m atmforecast.cli generate-data --atms 5 --days 1095
```

## 3. Methodology

### 3.1 Baselines (the yardstick)
Any serious model must beat these. Implemented from scratch in
`models/baselines.py`:

- **Mean**: $\hat y_{t+k}=\bar y$
- **Drift** (random walk with drift): $\hat y_{t+k}=y_t + k\frac{y_t-y_1}{t-1}$
- **Moving average** of the last $w$ days
- **Seasonal naïve**: $\hat y_{t+k}=y_{t+k-m}$ (repeat last week, $m=7$)

### 3.2 Holt–Winters triple exponential smoothing *(implemented from scratch)*
The mathematical core, in `models/holt_winters.py`. It tracks level $\ell_t$,
trend $b_t$ and seasonal $s_t$ states. Additive form:

$$
\begin{aligned}
\ell_t &= \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1}+b_{t-1})\\
b_t &= \beta(\ell_t-\ell_{t-1}) + (1-\beta)b_{t-1}\\
s_t &= \gamma(y_t-\ell_t) + (1-\gamma)s_{t-m}\\
\hat y_{t+h} &= \ell_t + h\,b_t + s_{t-m+((h-1)\bmod m)+1}
\end{aligned}
$$

with a multiplicative seasonal variant for demand whose swings scale with level.
The smoothing parameters $(\alpha,\beta,\gamma)\in[0,1]^3$ are fitted by
minimising the in-sample one-step SSE via a coarse grid search followed by
coordinate descent (no SciPy dependency).

### 3.3 SARIMA / SARIMAX (statsmodels)
`models/sarima.py`. Seasonal ARIMA in backshift-operator form,

$$
\Phi_P(B^m)\phi_p(B)(1-B)^d(1-B^m)^D y_t = \Theta_Q(B^m)\theta_q(B)\varepsilon_t + \beta^\top x_t,
$$

where $x_t$ are **exogenous calendar regressors** (salary-day and festival
dummies + Fourier terms from `features/calendar_features.py`) so the model can
*anticipate* the deterministic spikes a pure ARIMA would smear out. Order
selection via `auto_arima` (AICc); differencing justified by ADF + KPSS
stationarity tests (`check_stationarity`).

### 3.4 Prophet & LSTM
- **Prophet** (`models/prophet_model.py`): additive trend + Fourier
  seasonalities + an Indian-holiday component.
- **LSTM** (`models/lstm_model.py`): a Keras sequence model on standardised
  sliding windows, multi-step **recursive** forecasting, with **Monte-Carlo
  dropout** prediction intervals.

### 3.5 Evaluation — rolling-origin backtesting
A single train/test split is a single noisy sample. We use **rolling-origin
(walk-forward) cross-validation** (`evaluation/backtest.py`): slide the forecast
origin forward, refit on the past only, score the next $H$ days, and average.
Metrics (`evaluation/metrics.py`): MAE, RMSE, MAPE, sMAPE, and **MASE** (scaled
by the in-sample seasonal-naïve error; **MASE < 1 ⇒ beats seasonal naïve**).

## 4. Results (reproducible)

52-fold rolling-origin backtest, $H=14$, weekly step, 1-year minimum training
window, on the bundled dataset:

**ATM001 leaderboard (ranked by MASE):**

| model | MAE | RMSE | MAPE | MASE |
|---|---:|---:|---:|---:|
| **holt_winters** | 176,267 | 214,007 | 14.95% | **0.964** |
| seasonal_naive | 185,783 | 242,549 | 15.14% | 1.014 |
| mean | 281,138 | 348,296 | 23.32% | 1.532 |
| moving_average | 295,590 | 360,520 | 25.95% | 1.611 |
| drift | 343,656 | 397,243 | 32.76% | 1.876 |

**Best model per ATM:**

| ATM | best model | MASE | MAPE |
|---|---|---:|---:|
| ATM001 | holt_winters | 0.964 | 14.9% |
| ATM002 | holt_winters | 0.934 | 14.9% |
| ATM003 | holt_winters | 0.885 | 15.2% |
| ATM004 | holt_winters | 1.024 | 16.9% |
| ATM005 | holt_winters | 0.997 | 16.6% |

**Reading of the result.** Holt–Winters consistently beats every baseline: it
captures the dominant weekly cycle that the flat mean/moving-average cannot, and
adapts its level/trend better than the seasonal-naïve repeat. The remaining
error is driven largely by the *monthly* salary cycle and festivals, which are
aperiodic at the weekly scale — precisely the structure the SARIMAX / Prophet
calendar regressors are designed to absorb, and the natural next step when the
full scientific stack is available.

Regenerate all numbers with:

```bash
python -m atmforecast.cli backtest --atm ATM001
python scripts/run_pipeline.py            # full batch over all ATMs -> artifacts/
```

## 5. From forecast to decision — the cash-inventory layer

`operations/cash_optimizer.py` closes the loop. Given per-rupee under-stock
(stock-out) cost $C_u$ and over-stock (holding) cost $C_o$, the cost-optimal
service level is the **newsvendor critical ratio**

$$
p^\* = \frac{C_u}{C_u + C_o},
$$

and the cash to load for a replenishment cycle of length $L$ is the base-stock
level

$$
S = \underbrace{\textstyle\sum_{k=1}^{L}\hat y_{t+k}}_{\text{expected demand}} + \underbrace{z_{p^\*}\,\hat\sigma\sqrt{L}}_{\text{safety stock}},
$$

where $z_{p^\*}$ is the standard-normal quantile (computed from scratch via
Acklam's inverse-normal approximation) and $\hat\sigma$ is the one-day
forecast-error standard deviation. `simulate_policy` then costs any policy
against a realised demand path (replenishment + stock-out penalty + holding).

## 6. RAG analyst layer

Mirrors the FinApp RAG sample. The numerical artifacts are rendered into
Markdown **reports** (`rag/report_builder.py`), embedded with **Gemini
`text-embedding-004`** into **ChromaDB** (`rag/vectorstore.py`), and answered by
**Gemini** through a grounded **LangChain** prompt (`rag/agent.py`). External
documents (e.g. an RBI cash-management circular) can be ingested from **PDF**
via PyPDF. Example questions:

- *“How much cash should I load into ATM003 next week?”*
- *“Which ATM has the highest stock-out risk, and why?”*
- *“Compare the backtest accuracy of Holt-Winters vs seasonal naïve for ATM001.”*

## 7. Project structure

```
atm-cash-forecasting/
├── src/atmforecast/
│   ├── data/            synthetic generator + loaders
│   ├── features/        calendar & holiday regressors, Fourier terms
│   ├── models/          baselines, Holt-Winters (scratch), SARIMA, Prophet, LSTM
│   ├── evaluation/      metrics + rolling-origin backtest
│   ├── operations/      newsvendor cash-inventory optimiser
│   ├── rag/             report builder, ChromaDB store, Gemini LLM, agent
│   ├── api/             FastAPI app (sessions + SSE streaming)
│   ├── service.py       orchestration shared by CLI / API / pipeline
│   └── cli.py           `atmf` command-line interface
├── scripts/run_pipeline.py
├── tests/               52 unit tests (pure-Python core)
├── config/config.yaml   ├── Dockerfile ├── docker-compose.yml ├── Makefile
└── data/raw/atm_transactions.csv   (bundled synthetic dataset)
```

## 8. Installation & usage

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt

# 2. data + a forecast
atmf generate-data
atmf forecast  --atm ATM001 --model holt_winters --horizon 14
atmf backtest  --atm ATM001
atmf cash-plan --atm ATM001 --service-level 0.95 --balance 5000000

# 3. full batch pipeline -> artifacts/
python scripts/run_pipeline.py

# 4. RAG analyst (needs a Gemini key)
cp .env.example .env          # then add GOOGLE_API_KEY
atmf index --atm ATM001
atmf chat "How much cash should I load into ATM001 next week?"

# 5. serve the streaming API
atmf serve                    # or: make serve   /   docker compose up --build
```

Key API endpoints (`uvicorn atmforecast.api.app:app`): `POST /forecast`,
`POST /cash-plan`, `POST /index/{atm_id}`, `POST /ingest/pdf`, `POST /chat`,
`POST /chat/stream` (Server-Sent Events). Interactive docs at `/docs`.

### Testing

```bash
pytest -q          # 52 tests; the pure-Python core needs no third-party packages
```

## 9. Notes on the implementation

The data generator, evaluation metrics, all baselines, **Holt–Winters**, the
rolling-origin backtester and the cash-inventory optimiser are written in **pure
standard-library Python** and covered by unit tests — the mathematics is
transparent and dependency-free. The heavier models (SARIMA, Prophet, LSTM) and
the RAG/serving layers build on the scientific / LLM stack in
`requirements.txt`.

## 10. Limitations & future work

- Single-machine univariate models; a **hierarchical / global** model pooling
  the whole fleet would share strength across ATMs.
- The synthetic DGP is a stand-in for real transaction logs; swap in real data
  by matching the CSV schema.
- Extend the inventory layer to a **multi-echelon / routing** problem (which van
  visits which ATMs, and when) — the natural operations-research sequel.

## References (methods)

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (rolling-origin CV, MASE, ETS/Holt-Winters).
- Box, Jenkins, Reinsel, *Time Series Analysis: Forecasting and Control* (SARIMA).
- Taylor & Letham (2018), *Forecasting at Scale* (Prophet).
- Hyndman & Koehler (2006), *Another look at measures of forecast accuracy* (MASE).

*Content on external methods was paraphrased/summarised for compliance.*
