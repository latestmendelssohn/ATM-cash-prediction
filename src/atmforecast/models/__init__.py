"""
Forecasting models.

Pure-Python (stdlib only), verified in-sandbox:
    baselines.SeasonalNaive, baselines.MovingAverage, baselines.Drift, baselines.MeanForecast
    holt_winters.HoltWinters   (additive/multiplicative triple exponential smoothing)

Library-based (require the scientific stack in requirements.txt):
    sarima.SarimaModel         (statsmodels SARIMAX + optional pmdarima auto order)
    prophet_model.ProphetModel (Meta Prophet)
    lstm_model.LSTMForecaster  (Keras/TensorFlow sequence model)

All models implement the small ``BaseForecaster`` protocol: ``fit(y)`` then
``predict(h)`` returning a list of point forecasts (length ``h``).
"""

from .base import BaseForecaster, ForecastResult  # noqa: F401
from .baselines import Drift, MeanForecast, MovingAverage, SeasonalNaive  # noqa: F401
from .holt_winters import HoltWinters  # noqa: F401

# Registry used by the backtest harness / CLI to build models by name.
PURE_PYTHON_MODELS = {
    "mean": MeanForecast,
    "drift": Drift,
    "moving_average": MovingAverage,
    "seasonal_naive": SeasonalNaive,
    "holt_winters": HoltWinters,
}
