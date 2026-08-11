# Predicting ATM Cash Balance using Time Series

**A project that forecasts how much cash an ATM will hand out each day, is honest
about how uncertain that guess is, and uses both to decide how much money to load
into the machine — with a Gemini-powered RAG analyst.**

A project that forecasts daily ATM cash demand, quantify its uncertainty, and turn that into a cost-optimal cash-loading decision, with a Gemini-powered RAG analyst.

Every ATM loses money two ways: stock runs outs (the machine runs dry — lost customers) and idle cash (over-loaded machines tie up currency that earns nothing). Both come from not knowing tomorrow's demand. This project builds a time-series forecaster for daily withdrawals, attaches a 95% prediction interval, and converts it into a replenishment policy using inventory theory.


This project is deliberately small. All of the mathematics lives in **five
Python files**, and the parts that matter most run on the standard library
alone, so you can read them without wading through a framework.

| file | what lives here |
|---|---|
| `data.py` | a synthetic ATM data generator + simple loaders (pure standard library) |
| `models.py` | error metrics, baselines, **Holt-Winters from scratch**, SARIMA, backtesting, and the cash-planning optimiser |
| `analyst.py` | the RAG layer — turning results into short reports, embedding them, and answering questions with Gemini |
| `app.py` | a FastAPI streaming service **and** the command-line interface |
| `test_core.py` | tests for the core mathematics |

---

## The problem, stated precisely

Fix one machine. Let $y_t$ be the net cash it dispenses on day $t$ (withdrawals
minus the occasional deposit), measured in rupees. Given the history
$y_1,\dots,y_n$ we want three things:

1. a **point forecast** $\hat y_{n+1},\dots,\hat y_{n+H}$ for the next
   $H = 14$ days;
2. an honest **prediction interval** $[\hat\ell_{n+k},\,\hat u_{n+k}]$ around
   each of those days, at a stated confidence such as 95%; and
3. a **replenishment decision** — the amount $S$ of cash to load — chosen so
   that the chance of running dry before the next refill is acceptably small,
   *and* so that we are not needlessly hoarding cash.

The first two are statistics; the third is operations research. The whole point
of the project is that they belong together.

## How ATM cash demand actually behaves

Before choosing a model it pays to look at the data the way a person would. ATM
withdrawals are not random noise around a fixed number — they carry a lot of
predictable structure, most of it driven by the **calendar** rather than by the
machine's own recent past:

- a strong **weekly rhythm** — people pull out cash for the weekend, so
  Friday–Sunday run high and mid-week runs low;
- a **monthly salary-and-rent cycle** — a spike in the first few days of the
  month and again at month-end, when salaries land and bills fall due;
- **festival spikes** — Diwali, Christmas and the like, when cash spending jumps
  for a day or two;
- a slow **trend**, as a neighbourhood grows or card/UPI usage nibbles at cash.

A classical way to say this is that the series decomposes, roughly, into
*trend + seasonality + remainder*:

$$y_t \;=\; T_t \;+\; S_t \;+\; R_t,$$

where $T_t$ moves slowly, $S_t$ repeats with a period $m$ (here $m = 7$ days),
and $R_t$ is what is left over. A good forecaster is essentially a good way of
estimating $T_t$ and $S_t$ from noisy history and extrapolating them, while
treating $R_t$ as the irreducible uncertainty we must quantify rather than
predict.

One more idea worth naming is **stationarity**. Many classical tools assume the
statistical behaviour of a series (its mean, its variance) does not drift over
time. ATM demand is clearly *not* stationary as-is — it trends and it has
seasonality — so the standard remedy is *differencing*: modelling the change
$y_t - y_{t-1}$, or the seasonal change $y_t - y_{t-m}$, which removes trend and
seasonal level and leaves something much closer to stationary. That single idea
is the backbone of the ARIMA family below.

## My methods: the models and the thinking behind them

### Baselines — the honest yardstick

It is easy to be impressed by a complicated model until you check whether a
trivial one does just as well. So we always measure against two baselines:

- the **historical mean**, $\hat y_{n+k} = \bar y$ — the "nothing ever changes"
  forecast; and
- the **seasonal naïve**, $\hat y_{n+k} = y_{n+k-m}$ — literally "next Monday
  will look like last Monday".

The seasonal naïve is a surprisingly tough opponent precisely because so much of
the signal is weekly. If a fancy model cannot beat it, the fancy model is not
earning its keep.

### Holt-Winters — exponential smoothing, built from scratch

Exponential smoothing rests on a very human intuition: **recent observations
should count for more than old ones, but the old ones should not be thrown away
entirely.** A simple exponentially weighted average of the past does exactly
that. Holt-Winters extends the idea to series that also have a trend and a
season, by tracking three quantities that are updated a little bit each day:

- a **level** $\ell_t$ — where the series is right now;
- a **trend** $b_t$ — how fast it is drifting up or down;
- a **seasonal** term $s_t$ — how this particular day of the week differs from
  the level.

In additive form the update equations are

$$
\begin{aligned}
\ell_t &= \alpha\,(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1}),\\
b_t &= \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\,b_{t-1},\\
s_t &= \gamma\,(y_t - \ell_t) + (1-\gamma)\,s_{t-m},
\end{aligned}
$$

and the forecast simply carries the level and trend forward and adds back the
appropriate day-of-week term,

$$\hat y_{t+h} = \ell_t + h\,b_t + s_{t-m+((h-1)\bmod m)+1}.$$

Read those equations as *"nudge each belief towards what today's data suggests,
but only partway."* The three smoothing constants $\alpha,\beta,\gamma \in [0,1]$
control how reactive each belief is: near 1 the model trusts the latest day and
adapts fast (but jitters); near 0 it changes its mind slowly (but lags). Rather
than guess them, we **fit** $(\alpha,\beta,\gamma)$ by minimising the in-sample
one-step-ahead sum of squared errors, using a coarse grid search followed by a
few rounds of coordinate descent — enough to land near the optimum of this
smooth, low-dimensional objective without needing SciPy. There is also a
multiplicative-seasonal variant for machines whose weekly swings grow in
proportion to their overall level.

### SARIMA — modelling the correlations directly

Where Holt-Winters describes the series through evolving states, ARIMA describes
it through the **autocorrelation** of its shocks. Using the backshift operator
$B$ (defined by $B\,y_t = y_{t-1}$), a seasonal ARIMA model of order
$(p,d,q)(P,D,Q)_m$ is written compactly as

$$\Phi_P(B^m)\,\phi_p(B)\,(1-B)^d(1-B^m)^D\,y_t = \Theta_Q(B^m)\,\theta_q(B)\,\varepsilon_t.$$

It looks dense, but each piece has a plain meaning. The factors $(1-B)^d$ and
$(1-B^m)^D$ are the **differencing** that removes trend and seasonal level to
reach stationarity. The $\phi$ and $\Phi$ polynomials are the **autoregressive**
part — today depends on recent (and same-day-last-week) values. The $\theta$ and
$\Theta$ polynomials are the **moving-average** part — today depends on recent
random shocks. In practice one chooses the orders by inspecting the
autocorrelation (ACF) and partial-autocorrelation (PACF) plots and by comparing
candidate models on an information criterion such as the AIC, and one justifies
the amount of differencing with stationarity tests (ADF, KPSS). Here SARIMA is
available through `statsmodels` as an optional, heavier-weight alternative to
Holt-Winters.

## Being honest about uncertainty

A single number is a fragile thing to bet cash on, so every forecaster in this
project can also return a **prediction interval**. We estimate the standard
deviation $\hat\sigma$ of the one-step residuals and form a band

$$\hat y_{t+k} \;\pm\; z_{1-\alpha/2}\,\hat\sigma\,\sqrt{k},$$

where $z_{1-\alpha/2}$ is the familiar normal quantile (1.96 for 95%). The
$\sqrt{k}$ is the interesting part: it says our uncertainty *grows the further
out we forecast*, and it grows like the square root of the horizon. That is
exactly what you get when errors accumulate like a random walk — the variance of
a sum of $k$ independent shocks is $k$ times the one-step variance, so the
standard deviation scales as $\sqrt{k}$. In words: guessing tomorrow is easy,
guessing two weeks out is not, and the band should visibly fan out to admit it.

## Judging the models fairly

It is tempting to split the data once — train on the first stretch, test on the
last — and report the score. But that is one noisy sample, and it quietly leaks
information if you tune anything against it. The disciplined alternative for
time-ordered data is **rolling-origin evaluation** (walk-forward cross-
validation): pick an origin, train only on the past up to that point, forecast
the next $H$ days, score them, then slide the origin forward and repeat. On the
bundled data this yields **52 folds**, and averaging over them gives a far more
trustworthy ranking than any single split. Crucially, the model never sees the
future it is being tested on.

For the scores themselves we report several complementary metrics, because each
tells a slightly different story:

- **MAE** and **RMSE** — average error in rupees; RMSE punishes big misses harder.
- **MAPE** / **sMAPE** — the same idea as a percentage, so numbers are comparable
  across machines of very different sizes.
- **MASE** (mean absolute scaled error) — the one to trust most. It divides our
  error by the error a seasonal-naïve forecast would have made *in-sample*, so it
  is scale-free and has a clean reading: **MASE below 1 means we beat the naïve
  benchmark; above 1 means we did not.** It also behaves gracefully when some
  actuals are zero, where MAPE blows up.

## Turning a forecast into a decision

This is where forecasting stops being an academic exercise. Suppose we must load
$S$ rupees to cover the next cycle, and demand $D$ over that cycle is uncertain.
Every rupee we are short costs us $C_u$ (a stock-out — lost goodwill and
penalties); every rupee left idle costs us $C_o$ (carrying and opportunity cost).
The expected cost is

$$C(S) = C_o\,\mathbb{E}\big[(S-D)^+\big] + C_u\,\mathbb{E}\big[(D-S)^+\big].$$

Differentiating and setting the derivative to zero gives the classic
**newsvendor** result: the optimal load is the demand quantile at the
*critical ratio*

$$F(S^\*) = \frac{C_u}{C_u + C_o}\;=:\;p^\*,$$

where $F$ is the demand distribution. In plain terms, the cheaper a stock-out is
relative to holding cash, the lower the service level you should aim for, and
vice versa — the formula just makes that trade-off exact. Modelling cycle demand
as approximately normal, $S^\*$ becomes

$$S^\* = \underbrace{\sum_{k=1}^{L}\hat y_{n+k}}_{\text{expected demand}} \;+\; \underbrace{z_{p^\*}\,\hat\sigma\sqrt{L}}_{\text{safety stock}}.$$

The first term is what we expect to pay out; the second is a **safety-stock**
buffer sized to our uncertainty and our appetite for risk. The normal quantile
$z_{p^\*}$ is computed from scratch (Acklam's rational approximation to the
inverse normal CDF), so this part, too, needs no third-party libraries. Given a
machine's current balance, the tool then reports the concrete top-up to send on
the next van.

## Asking questions in plain English

Numbers in a JSON file rarely reach the person making the call. The RAG
(retrieval-augmented generation) layer bridges that gap. Each forecast, backtest
leaderboard and cash plan is rendered into a short Markdown **report**; those
reports are embedded with Gemini and stored in a ChromaDB vector index; and when
you ask a question, the most relevant reports are retrieved and handed to Gemini
as grounded context before it answers. You can also feed in an external **PDF**
— say, an internal cash-management circular — and it becomes part of what the
analyst can draw on. Because the answer is built from retrieved facts rather than
the model's memory, it stays anchored to *your* numbers.

## Results — reproducible on the bundled data

A 52-fold rolling-origin backtest at $H = 14$ tells a clean story: **Holt-Winters
wins on every machine.**

| ATM | best model | MASE |
|---|---|---:|
| ATM001 | holt_winters | 0.976 |
| ATM002 | holt_winters | 0.977 |
| ATM003 | holt_winters | 0.865 |
| ATM004 | holt_winters | 0.994 |
| ATM005 | holt_winters | 0.921 |

For ATM001 the full leaderboard reads holt_winters (MASE **0.976**) ahead of
seasonal-naïve (1.083) and the historical mean (1.572). The reading is intuitive:
the mean ignores the weekly rhythm entirely and pays for it, the seasonal-naïve
captures the rhythm but cannot adapt its level or trend, and Holt-Winters does
both. What error remains is largely the *monthly* salary cycle and festivals,
which are aperiodic at the weekly scale — exactly the structure that exogenous
calendar regressors in a SARIMAX model are built to absorb, and the natural next
step for anyone extending this work.

Reproduce it yourself:

```bash
python app.py backtest --atm ATM001     # one machine's leaderboard
python app.py pipeline                  # every machine at once
```

## Project layout

```
data.py         synthetic data generation + loaders (pure stdlib)
models.py       metrics · baselines · Holt-Winters · SARIMA · backtest · cash plan
analyst.py      RAG: reports + ChromaDB + Gemini + LangChain
app.py          FastAPI streaming API + CLI
test_core.py    tests for the core maths
data/atm_transactions.csv   bundled synthetic dataset (5 ATMs × 3 years)
```

## Quickstart

```bash
pip install -r requirements.txt          # the core actually runs without this

python data.py                           # (re)generate the dataset
python app.py forecast  --atm ATM001     # 14-day forecast + interval
python app.py backtest  --atm ATM001     # model leaderboard
python app.py cash-plan --atm ATM001 --balance 1.5e7
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

## A note on the data

Real ATM transaction logs are confidential, so the bundled dataset is generated
by a process with *known* structure (`data.py`) — which is rather convenient for
teaching, since we can check whether each model recovers the seasonality and
trend we planted. To use this on real data, match the CSV schema
(`date, atm_id, net_cash_out, …`) and everything downstream just works.

The generator, metrics, baselines, Holt-Winters, backtester and cash optimiser
are **pure standard-library Python** (no numpy or pandas) and are covered by
`test_core.py`. SARIMA, the RAG analyst and the API build on the packages listed
in `requirements.txt`.

## References

- Hyndman, R. J. & Athanasopoulos, G. *Forecasting: Principles and Practice* —
  exponential smoothing / ETS, rolling-origin cross-validation, MASE.
- Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. *Time Series Analysis:
  Forecasting and Control* — the ARIMA / SARIMA framework.
- Hyndman, R. J. & Koehler, A. B. (2006). *Another look at measures of forecast
  accuracy* — the case for MASE.
- The newsvendor model is standard inventory theory; see any operations-research
  text for the critical-ratio derivation.

*The descriptions of these methods are paraphrased and summarised in my own
words for licensing compliance.*
