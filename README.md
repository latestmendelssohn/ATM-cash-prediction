# Predicting ATM cash balance using time series

This project forecasts how much cash an ATM will dispense each day, quantifies the
uncertainty in that forecast, and uses both to decide how much money to load into
the machine. A Gemini-powered RAG analyst sits on top so the results can be queried
in plain English.

An ATM loses money in two ways. It can run dry, which costs customers, or it can sit
over-loaded, so that currency is tied up earning nothing. Both problems come from not
knowing tomorrow's demand. So the project builds a time-series forecaster for daily
withdrawals, attaches a 95% prediction interval to it, and converts that into a
replenishment policy using inventory theory.

| file | what lives here |
|---|---|
| `data.py` | a synthetic ATM data generator plus simple loaders (pure standard library) |
| `models.py` | error metrics, baselines, Holt-Winters from scratch, SARIMA, backtesting, and the cash-planning optimiser |
| `analyst.py` | the RAG layer: turning results into short reports, embedding them, and answering questions with Gemini |
| `app.py` | a FastAPI streaming service and the command-line interface |
| `tests/test_core.py` | tests for the core mathematics |

---

## The problem, stated precisely

Fix one machine. Let $y_t$ be the net cash it dispenses on day $t$ (withdrawals
minus the occasional deposit), measured in rupees. Given the history
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
slide the origin forward and repeat. On the bundled data this yields 52 folds, and
averaging over them gives a far more trustworthy ranking than any single split. The
model never sees the future it is being tested on.

For the scores themselves we report several complementary metrics, because each tells
a slightly different story:

- MAE and RMSE, the average error in rupees, where RMSE punishes big misses harder.
- MAPE and sMAPE, the same idea as a percentage, so numbers are comparable across
  machines of very different sizes.
- MASE, the mean absolute scaled error, and the one to trust most. It divides our
  error by the error a seasonal naive forecast would have made in-sample, so it is
  scale-free and reads cleanly: below 1 means we beat the naive benchmark, above 1
  means we did not. It also behaves gracefully when some actuals are zero, where MAPE
  blows up.

## Turning a forecast into a decision

Suppose we must load $S$ rupees to cover the next cycle, and demand $D$ over that
cycle is uncertain. Every rupee we are short costs us $C_u$, a stock-out with lost
goodwill and penalties. Every rupee left idle costs us $C_o$, the carrying and
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
$L$-day total). For ATM001 that measured spread is Rs 23.30 lakh against the Rs 6.68
lakh the independent-errors formula implies, which raises the 95% safety stock from
about Rs 11.0 lakh to Rs 38.3 lakh. Measuring the error you care about is worth more
here than any refinement of the point forecast.

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

## Results, reproducible on the bundled data

A 52-fold rolling-origin backtest at $H = 14$ gives a consistent answer:
Holt-Winters wins on every machine.

| ATM | best model | MASE |
|---|---|---:|
| ATM001 | holt_winters | 0.976 |
| ATM002 | holt_winters | 0.977 |
| ATM003 | holt_winters | 0.865 |
| ATM004 | holt_winters | 0.994 |
| ATM005 | holt_winters | 0.921 |

For ATM001 the full leaderboard, all four models on the same 52 folds, reads:

| model | MAE | RMSE | MAPE | sMAPE | MASE |
|---|---:|---:|---:|---:|---:|
| holt_winters | 169,759 | 210,827 | 14.15 | 13.97 | 0.976 |
| sarima | 183,808 | 225,073 | 15.55 | 15.23 | 1.057 |
| seasonal_naive | 188,557 | 246,192 | 15.34 | 15.07 | 1.083 |
| mean | 273,377 | 336,059 | 22.58 | 22.83 | 1.572 |

That ordering makes sense. The mean ignores the weekly rhythm entirely and pays for
it, the seasonal naive forecast captures the rhythm but cannot adapt its level or
trend, and Holt-Winters does both. SARIMA, at the fixed order $(1,1,1)(1,1,1)_7$ with
no tuning per fold, lands between them: better than repeating last week, worse than
smoothing. What error remains is largely the monthly salary cycle and the festivals,
which are aperiodic at the weekly scale. That is exactly the structure exogenous
calendar regressors in a SARIMAX model are built to absorb, and it is the natural
next step for anyone extending this work.

One caveat about what is being forecast. The generator records
`dispensed = min(demand, balance)`, so on a stock-out day the series holds the cash
the machine could pay out, not the demand that walked up to it. Demand is censored
from above exactly when it is highest, which biases any model fitted to it downward.
The bundled data is configured so stock-outs are rare, but the point matters for real
transaction logs.

Reproduce it yourself:

```bash
python app.py backtest --atm ATM001            # the three stdlib models
python app.py backtest --atm ATM001 --sarima   # adds SARIMA (slow, needs statsmodels)
python app.py pipeline                         # every machine at once
```

## Project layout

```
data.py         synthetic data generation + loaders (pure stdlib)
models.py       metrics · baselines · Holt-Winters · SARIMA · backtest · cash plan
analyst.py      RAG: reports + ChromaDB + Gemini + LangChain
app.py          FastAPI streaming API + CLI
tests/          tests for the core maths
data/atm_transactions.csv   bundled synthetic dataset (5 ATMs × 3 years)
```

## Quickstart

```bash
pip install -r requirements.txt          # the core actually runs without this

python data.py                           # (re)generate the dataset
python app.py forecast  --atm ATM001     # 14-day forecast + interval
python app.py backtest  --atm ATM001     # model leaderboard
python app.py cash-plan --atm ATM001 --balance 1.5e7
python app.py cash-plan --atm ATM001 --cu 9 --co 1   # service level from the costs
python app.py pipeline                   # forecast + plan for every ATM
pytest                                   # run the core tests
```

### The RAG analyst (needs a Gemini key)

```bash
cp .env.example .env        # then add your GOOGLE_API_KEY
python app.py index --atm ATM001
python app.py chat "How much cash should I load into ATM001 next week?"
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

## A note on the data

The bundled dataset is generated by `data.py`.

The generator, metrics, baselines, Holt-Winters, backtester and cash optimiser use
the Python standard library only, with no numpy or pandas, and they are covered by
`test_core.py`. SARIMA, the RAG analyst and the API build on the packages listed in
`requirements.txt`.
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
