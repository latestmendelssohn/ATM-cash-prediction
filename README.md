# Predicting ATM cash balance using time series

[Open the live Streamlit demo](https://atm-cash-prediction-5n2wgq9gbhfm7sr25q3zfg.streamlit.app/)

This project forecasts how much cash an ATM will dispense each day, quantifies the
uncertainty in that forecast, and uses both to decide how much money to load into
the machine. A Gemini-powered RAG analyst sits on top so the results can be queried
in plain English.

The public demo is the small Streamlit app. It uses the bundled 2009-2010 public
ATM dataset, Holt-Winters, prediction intervals and cash planning only. It does not
use Gemini, store credentials, or connect to a bank system. The API, SARIMA model
and RAG analyst remain available for local use.

## Project at a glance

| part | entry point | purpose |
|---|---|---|
| Core | `data.py`, `models.py` | Generate/load data, forecast, backtest and calculate cash plans |
| API and CLI | `app.py` | FastAPI endpoints and command-line workflows |
| RAG analyst | `analyst.py` | Optional Gemini and ChromaDB reports and question answering |
| Hosted demo | `streamlit_app.py` | Public Streamlit Community Cloud interface |
| Alternative demo | `gradio_app.py` | Local Gradio interface using the same demo logic |
| Shared demo code | `demo_logic.py` | One forecast/cash-plan callback used by both UIs |
| Tests | `tests/` | Core mathematics and API smoke tests |

The main data flow is:

```text
data/raw/ATMData.csv  --(data_preprocess.py)-->  data/processed/atm_daily.csv
                                                        |
                                                        v
                                                    models.py  --->  app.py (CLI + API)
                                                        |          ---> analyst.py (optional Gemini RAG)
                                                        +---------> streamlit_app.py / gradio_app.py
```

The project is a student research and demonstration project, not a production
cash-management system.

---

## The problem, stated precisely

Fix one machine. Let $y_t$ be the net cash it dispenses on day $t$ (withdrawals
minus the occasional deposit), in the source unit of the dataset. The bundled
public data does not identify a currency, so all cash quantities in this project
are printed in that unit without a symbol. Given the history
$y_1,\dots,y_n$ we want three things:

1. A point forecast $\hat y_{n+1},\dots,\hat y_{n+H}$ for the next $H = 14$ days.
2. A prediction interval $[\hat\ell_{n+k},\,\hat u_{n+k}]$ around each of those
   days, at 95% confidence.
3. A replenishment decision: the amount $S$ of cash to load, chosen so that the
   chance of running dry before the next refill is under 5%, and so that we are not
   needlessly hoarding cash.

## How ATM cash demand actually behaves

Before choosing a model it pays to look at the data the way a person would. ATM
withdrawals are not random noise around a fixed number. They carry a lot of
predictable structure, and most of it is driven by the calendar rather than by the
machine's own recent past:

- A strong weekly rhythm. People pull out cash for the weekend, so Friday to Sunday
  run high and mid-week runs low.
- A monthly salary-and-rent cycle, with a spike in the first few days of the month
  and again at month end, when salaries land and bills fall due.
- Festival spikes: Diwali, Christmas and the like, when cash spending jumps for a day
  or two.
- A slow trend, as a neighbourhood grows or as card and UPI usage nibbles at cash.

One way to say this is that the series decomposes, roughly, into trend plus
seasonality plus a remainder:

$$y_t \;=\; T_t \;+\; S_t \;+\; R_t,$$

where $T_t$ moves slowly, $S_t$ repeats with a period $m$ (here $m = 7$ days), and
$R_t$ is what is left over. A good forecaster is essentially a good way of estimating
$T_t$ and $S_t$ from noisy history and extrapolating them, while treating $R_t$ as
the irreducible uncertainty we have to quantify rather than predict.

One more idea worth naming is stationarity. Many classical tools assume the
statistical behaviour of a series (its mean, its variance) does not drift over time.
ATM demand is clearly not stationary as it stands, because it trends and it has
seasonality. The standard remedy is differencing: modelling the change
$y_t - y_{t-1}$, or the seasonal change $y_t - y_{t-m}$, which removes trend and
seasonal level and leaves something much closer to stationary.

## The models and the thinking behind them

### Benchmarks

Every model is measured against two benchmarks:

- the historical mean, $\hat y_{n+k} = \bar y$, which is the "nothing ever changes"
  forecast, and
- the seasonal naive forecast, $\hat y_{n+k} = y_{n+k-m}$, which says literally
  "next Monday will look like last Monday".

### Holt-Winters: exponential smoothing, built from scratch

Exponential smoothing rests on one idea: recent observations should count for more
than old ones, but the old ones should not be thrown away entirely. An exponentially
weighted moving average does that job well, and Holt-Winters extends it to series
that also have a trend and a season by tracking three quantities that are updated a
little bit each day:

- the level $\ell_t$, where the series is right now;
- the trend $b_t$, how fast it is drifting up or down;
- the seasonal term $s_t$, how this particular day of the week differs from the level.

In additive form the update equations are

$$
\begin{aligned}
\ell_t &= \alpha\,(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1}),\\
b_t &= \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\,b_{t-1},\\
s_t &= \gamma\,(y_t - \ell_t) + (1-\gamma)\,s_{t-m},
\end{aligned}
$$

and the forecast carries the level and trend forward, then adds back the appropriate
day-of-week term,

$$\hat y_{t+h} = \ell_t + h\,b_t + s_{t-m+((h-1)\bmod m)+1}.$$

Each equation says the same thing in a different place: move towards what today's
data suggests, but only partly. The three smoothing constants
$\alpha,\beta,\gamma \in [0,1]$ control how reactive each belief is, and near 1 the
model trusts the latest day and adapts fast. Rather than guess them, the code fits
$(\alpha,\beta,\gamma)$ by minimising the in-sample one-step-ahead sum of squared
errors, using a coarse grid search followed by a few rounds of coordinate descent.
That is enough to land near the optimum of this smooth, low-dimensional objective.

### SARIMA

ARIMA describes the series through the autocorrelation of its shocks. Using the
backshift operator $B$ (defined by $B\,y_t = y_{t-1}$), a seasonal ARIMA model of
order $(p,d,q)(P,D,Q)_m$ is written compactly as

$$\Phi_P(B^m)\,\phi_p(B)\,(1-B)^d(1-B^m)^D\,y_t = \Theta_Q(B^m)\,\theta_q(B)\,\varepsilon_t.$$

The factors $(1-B)^d$ and $(1-B^m)^D$ are the differencing that removes trend and
seasonal level to reach stationarity. The $\phi$ and $\Phi$ polynomials are the
autoregressive part, where today depends on recent values and on the same day last
week. The $\theta$ and $\Theta$ polynomials are the moving-average part, where today
depends on recent random shocks. In practice you choose the orders by inspecting the
autocorrelation (ACF) and partial autocorrelation (PACF) plots and by comparing
candidate models on an information criterion such as the AIC, and you justify the
amount of differencing with stationarity tests (ADF, KPSS). Here SARIMA is available
through `statsmodels` as an optional, heavier-weight alternative to Holt-Winters.

## Measuring the uncertainty

Every forecaster in this project returns a prediction interval. We estimate the
standard deviation $\hat\sigma$ of the one-step residuals and form a band

$$\hat y_{t+k} \;\pm\; z_{1-\alpha/2}\,\hat\sigma\,\sqrt{k},$$

where $z_{1-\alpha/2}$ is the familiar normal quantile, 1.96 for 95%. The $\sqrt{k}$
is the interesting part. It says our uncertainty grows the further out we forecast,
and that it grows like the square root of the horizon. That is exactly what you get
when errors accumulate like a random walk, because the variance of a sum of $k$
independent shocks is $k$ times the one-step variance, so the standard deviation
scales as $\sqrt{k}$. Guessing tomorrow is easy, guessing two weeks out is not, and
the band should visibly fan out to admit it.

## How the models are judged

It is tempting to split the data once, train on the first stretch, test on the last,
and report the score. But that is one noisy sample, and it quietly leaks information
if you tune anything against it. The disciplined alternative for time-ordered data is
rolling-origin evaluation, also called walk-forward cross-validation: pick an origin,
train only on the past up to that point, forecast the next $H$ days, score them, then
slide the origin forward and repeat. On the bundled real data, with a 180-day
minimum training window and a 14-day step, this yields 13 folds per ATM. Averaging
over the folds gives a far more trustworthy ranking than any single split. The
model never sees the future it is being tested on.

For the scores themselves we report several complementary metrics, because each tells
a slightly different story:

- MAE and RMSE, the average error in the source unit, where RMSE punishes big
  misses harder.
- MAPE and sMAPE, the same idea as a percentage, so numbers are comparable across
  machines of very different sizes.
- MASE, the mean absolute scaled error, and the one to trust most. It divides our
  error by the error a seasonal naive forecast would have made in-sample, so it is
  scale-free and reads cleanly: below 1 means we beat the naive benchmark, above 1
  means we did not. It also behaves gracefully when some actuals are zero, where MAPE
  blows up.

## Turning a forecast into a decision

Suppose we must load $S$ units of cash to cover the next cycle, and demand $D$ over
that cycle is uncertain. Every unit we are short costs us $C_u$, a stock-out with
lost goodwill and penalties. Every unit left idle costs us $C_o$, the carrying and
opportunity cost. The expected cost is

$$C(S) = C_o\,\mathbb{E}\big[(S-D)^+\big] + C_u\,\mathbb{E}\big[(D-S)^+\big].$$

Differentiating and setting the derivative to zero gives the classic newsvendor
result: the optimal load is the demand quantile at the critical ratio

$$F(S^\*) = \frac{C_u}{C_u + C_o}\;=:\;p^\*,$$

where $F$ is the demand distribution. In plain terms, the cheaper a stock-out is
relative to holding cash, the lower the service level you should aim for, and the
formula makes that trade-off exact. Modelling cycle demand as approximately normal,
$S^\*$ becomes

$$S^\* = \underbrace{\sum_{k=1}^{L}\hat y_{n+k}}_{\text{expected demand}} \;+\; \underbrace{z_{p^\*}\,\hat\sigma\sqrt{L}}_{\text{safety stock}}.$$

The first term is what we expect to pay out. The second is a safety-stock buffer
sized to our uncertainty and our appetite for risk. The normal quantile $z_{p^\*}$ is
computed from scratch, using Acklam's rational approximation to the inverse normal
CDF, so this part needs no third-party libraries either. Given a machine's current
balance, the tool then reports the concrete top-up to send on the next van.

Pass the two costs and the service level stops being a number picked by hand:
`--cu 9 --co 1` sets $p^*$ to 0.9 from the critical ratio.

One subtlety decides whether the safety stock is right. The per-day interval widens
like $\hat\sigma\sqrt{k}$, which assumes errors accumulate as the horizon grows. The
safety stock needs something different, the spread of the *total* over the whole
cycle, and $\hat\sigma\sqrt{L}$ only gives that if daily errors are independent. The
two assumptions cannot both hold, so rather than pick one, the code measures the
quantity it actually needs. `cycle_sigma_from_backtest` reuses the rolling-origin
folds and takes the standard deviation of (actual $L$-day total minus forecast
$L$-day total). On the bundled real ATM4 series the measured 14-day cycle spread is
3,207 in the source unit, while the independent-errors formula, using the model's
one-day residual sigma, implies a much smaller number. Cash-planning against the
measured spread raises the 95% safety stock to 5,275, an amount the daily-interval
shortcut would understate. Measuring the error you care about is worth more here
than any refinement of the point forecast.

## Asking questions in plain English

Numbers in a JSON file rarely reach the person making the call, and the RAG
(retrieval-augmented generation) layer is there to bridge that gap. Each forecast,
backtest leaderboard and cash plan is rendered into a short Markdown report. Those
reports are embedded with Gemini and stored in a ChromaDB vector index. When you ask
a question, the most relevant reports are retrieved and handed to Gemini as grounded
context before it answers. You can also feed in an external PDF, say an internal
cash-management circular, and it becomes part of what the analyst can draw on.
Because the answer is built from retrieved facts rather than the model's memory, it
stays anchored to your numbers.

## Results on the bundled real data

The bundled data is a small public ATM dataset with four machines and daily
withdrawals from 2009-05-01 to 2010-04-30. ATM3 is 362 zero-withdrawal days out
of 365 and is excluded from the leaderboard because it has no signal to forecast.
A 13-fold rolling-origin backtest at $H = 14$ (180-day minimum train, 14-day step)
gives:

| ATM | best model | MASE |
|---|---|---:|
| ATM1 | holt_winters | 0.91 |
| ATM2 | seasonal_naive | 0.79 |
| ATM4 | mean | 0.91 |

Unlike a longer, cleaner series, real 12-month data does not hand Holt-Winters
every machine. ATM2's best model is the seasonal naive forecast; ATM1 is
Holt-Winters by a comfortable margin; ATM4's average-based baseline wins because
a single outlier of 10,920 (against a median of 404) inflates the residuals of
any model that tries to fit it.

Full leaderboard for ATM1:

| model | MAE | RMSE | sMAPE | MASE |
|---|---:|---:|---:|---:|
| holt_winters | 17.70 | 23.34 | 28.92 | 0.91 |
| seasonal_naive | 19.70 | 25.78 | 28.52 | 1.01 |
| mean | 25.58 | 35.36 | 38.90 | 1.32 |

MAPE is dropped from the ATM2 table because ATM2 has zero-withdrawal days, and
zero actuals blow MAPE up. sMAPE and MASE handle zeros gracefully and are the
metrics to trust on this series.

At 95% service level the measured cycle-total spread from the same rolling
folds gives these recommended loads, in the source unit:

| ATM | forecast total (14d) | measured cycle sigma | safety stock | cycle load |
|---|---:|---:|---:|---:|
| ATM1 | 1,053 | 115 | 190 | 1,243 |
| ATM2 | 916 | 78 | 128 | 1,044 |
| ATM4 | 6,508 | 3,207 | 5,275 | 11,783 |

The interval coverage measured on the same folds is conservative at nominal 95%:
ATM1 99.4%, ATM2 98.9%, ATM4 99.4%. Numbers to reproduce these tables are in
`reports/real_backtest.md`, `reports/real_cash_policy.md`, and
`reports/real_interval_coverage.md`.

A few limitations of these numbers are worth stating up front:

- The source CSV does not identify a currency, so all cash quantities here are
  in the source unit without a symbol.
- The one-year source limits both the training window and the number of folds.
- ATM4's 10,920 observation is retained. It should be treated as a data-quality
  decision, not silently trimmed after seeing the scores.
- ATM3 is excluded from modelling; it is kept in the raw file for transparency.

Reproduce these tables yourself:

```bash
python app.py backtest --atm ATM1            # the three stdlib models
python app.py backtest --atm ATM1 --sarima   # adds SARIMA (slow, needs statsmodels)
python app.py pipeline                       # every usable machine at once
py -3.12 real_report.py                      # regenerate the full real-data reports
```

## Project layout

```
data.py             loaders + offline synthetic generator (pure stdlib)
data_loader.py      CSV parsing for the public source file
data_preprocess.py  raw -> processed daily rows with calendar features
models.py           metrics · baselines · Holt-Winters · SARIMA · backtest · cash plan
analyst.py          RAG: reports + ChromaDB + Gemini + LangChain
app.py              FastAPI streaming API + CLI
demo_logic.py       shared forecast + cash-plan logic for both demos
streamlit_app.py    Streamlit Community Cloud dashboard
gradio_app.py       Gradio alternative entry point
tests/              tests for the core maths and CSV pipeline
data/raw/           public source CSV (gitignored)
data/processed/     processed daily CSV (gitignored)
data/atm_transactions.csv   optional synthetic dataset (5 ATMs, 3 years)
```

## Quickstart

```bash
pip install -r requirements.txt          # the core actually runs without this

python data_preprocess.py --input data/raw/ATMData.csv --output data/processed/atm_daily.csv
python app.py forecast  --atm ATM1       # 14-day forecast + interval
python app.py backtest  --atm ATM1       # model leaderboard
python app.py cash-plan --atm ATM1 --balance 1.5e7
python app.py cash-plan --atm ATM1 --cu 9 --co 1     # service level from the costs
python app.py pipeline                   # forecast + plan for every usable ATM
py -3.12 real_report.py                  # regenerate the real-data reports
python data.py                           # (optional) regenerate the synthetic CSV
pytest                                   # run the core tests
```

### The RAG analyst (needs a Gemini key)

```bash
cp .env.example .env        # then add your GOOGLE_API_KEY
python app.py index --atm ATM1
python app.py chat "How much cash should I load into ATM1 next week?"
```

### The API

```bash
uvicorn app:app --reload
# or:  python app.py serve
# or:  docker build -t atmf . && docker run -p 8000:8000 atmf
```

Endpoints: `POST /forecast`, `POST /cash-plan`, `POST /index/{atm_id}`,
`POST /ingest/pdf`, `POST /chat`, `POST /chat/stream` (server-sent events).
Interactive docs live at `/docs`.

The API has open CORS and no authentication, which is fine on localhost and not
fine anywhere else, because `/chat` and `/index` spend Gemini quota per request.
Put it behind auth before exposing it on a network.

### Local Streamlit and Gradio demos

The project has two small UI entry points that share `demo_logic.py`. Both use only
the bundled real ATM dataset, Holt-Winters, prediction intervals and cash planning.
Neither uses Gemini or requires an API key.

```bash
pip install -r demo_requirements.txt

# Streamlit dashboard
streamlit run streamlit_app.py

# Gradio alternative
python gradio_app.py
```

### Deploying the Streamlit demo

The live demo runs on Streamlit Community Cloud from the `main` branch. To deploy
your own copy, open [share.streamlit.io](https://share.streamlit.io/), choose this
GitHub repository, select `main`, and set the main file to `streamlit_app.py`.
The root `requirements.txt` includes the Streamlit dependency used by the hosted app.

The hosted demo needs no secrets. It uses the committed processed CSV and does not
call Gemini. The Gradio entry point remains useful for local demos or a separate
Python host; Streamlit Community Cloud runs the Streamlit entry point only.

## A note on the data

The bundled dataset lives at `data/processed/atm_daily.csv`, built from the raw
public source at `data/raw/ATMData.csv` by `data_preprocess.py`. `data.py` also
ships an offline synthetic generator; running `python data.py` writes it to
`data/atm_transactions.csv` for experimentation, but the app, demos, and tests
run on the processed public data by default.

The loaders, preprocessing, metrics, baselines, Holt-Winters, backtester and cash
optimiser use the Python standard library only, with no numpy or pandas, and they
are covered by `test_core.py`. SARIMA, the RAG analyst and the API build on the
packages listed in `requirements.txt`.
## Limitations and scope

- The public source CSV does not identify a currency; all values are shown in
  the source unit without a symbol.
- The one-year source (2009-05-01 to 2010-04-30) limits both the training window
  and the number of rolling folds (13 per ATM at $H = 14$).
- ATM3 is excluded from modelling because 362 of its 365 days are zeros. It is
  kept in the raw file for transparency.
- ATM4's 10,920 maximum is retained. It should be treated as a data-quality
  decision, not silently removed after seeing the scores.
- MAPE is unreliable for ATM2 because zero-withdrawal days blow it up. MASE and
  sMAPE are the metrics to trust on that series.
- Holt-Winters models a weekly season. Monthly salary cycles and festival effects
  are not explicit regressors here; SARIMAX with calendar covariates is a natural
  next step.
- Daily prediction intervals use a widening normal approximation. Cash planning
  uses the measured spread of total cycle forecast errors from rolling-origin
  backtests.
- The API has open CORS and no authentication. Use it on localhost unless you add
  access control and quota protection.
- Gemini, ChromaDB and PDF ingestion are optional local features. The public
  Streamlit demo does not use them and does not require an API key.

## References

- Hyndman, R. J. & Athanasopoulos, G. *Forecasting: Principles and Practice*, for
  exponential smoothing and ETS, rolling-origin cross-validation, and MASE.
- Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. *Time Series Analysis: Forecasting
  and Control*, for the ARIMA and SARIMA framework.
- Hyndman, R. J. & Koehler, A. B. (2006). *Another look at measures of forecast
  accuracy*, for the case for MASE.
- The newsvendor model is standard inventory theory. See any operations-research text
  for the critical-ratio derivation.

The descriptions of these methods are my own summaries of the references above.
