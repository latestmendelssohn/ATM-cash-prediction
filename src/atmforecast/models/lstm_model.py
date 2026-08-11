r"""
LSTM sequence forecaster (Keras / TensorFlow).
=============================================

A Long Short-Term Memory network learns non-linear temporal dependencies that
the linear SARIMA cannot. We frame one-step-ahead prediction as supervised
learning on sliding windows of length ``lookback``:

    input  X_i = (z_{i}, ..., z_{i+L-1})        (L = lookback, scaled)
    target y_i =  z_{i+L}

Multi-step forecasts are produced **recursively**: each prediction is appended
to the input window and fed back in. The series is standardised to zero mean /
unit variance before training (LSTMs are sensitive to scale) and de-scaled on
output. Prediction intervals are obtained via **Monte-Carlo dropout** -- keeping
dropout active at inference and sampling several forward passes -- which gives a
cheap Bayesian-flavoured uncertainty estimate.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


class LSTMForecaster:
    def __init__(
        self,
        lookback: int = 28,
        units: int = 32,
        dropout: float = 0.1,
        epochs: int = 60,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        mc_samples: int = 50,
        seed: int = 42,
    ) -> None:
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples
        self.seed = seed
        self._model = None
        self._mu = 0.0
        self._sigma = 1.0
        self._history: List[float] = []

    # ------------------------------------------------------------- helpers
    def _standardise(self, y):
        import numpy as np

        arr = np.asarray(y, dtype="float32")
        self._mu = float(arr.mean())
        self._sigma = float(arr.std() + 1e-8)
        return (arr - self._mu) / self._sigma

    def _make_windows(self, z):
        import numpy as np

        X, Y = [], []
        for i in range(len(z) - self.lookback):
            X.append(z[i : i + self.lookback])
            Y.append(z[i + self.lookback])
        X = np.asarray(X).reshape(-1, self.lookback, 1)
        Y = np.asarray(Y).reshape(-1, 1)
        return X, Y

    def _build(self):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.random.set_seed(self.seed)
        net = models.Sequential(
            [
                layers.Input(shape=(self.lookback, 1)),
                layers.LSTM(self.units, return_sequences=False),
                layers.Dropout(self.dropout),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        )
        net.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mse",
        )
        return net

    # ---------------------------------------------------------------- API
    def fit(self, y: Sequence[float]) -> "LSTMForecaster":
        if len(y) <= self.lookback + 1:
            raise ValueError("series shorter than lookback window")
        self._history = list(map(float, y))
        z = self._standardise(y)
        X, Y = self._make_windows(z)
        self._model = self._build()
        self._model.fit(
            X, Y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=False,
        )
        return self

    def _recursive_forecast(self, h: int, training_flag: bool):
        import numpy as np

        z = list((np.asarray(self._history, dtype="float32") - self._mu) / self._sigma)
        window = z[-self.lookback :]
        preds = []
        for _ in range(h):
            x = np.asarray(window[-self.lookback :]).reshape(1, self.lookback, 1)
            yhat = float(self._model(x, training=training_flag).numpy().ravel()[0])
            preds.append(yhat)
            window.append(yhat)
        # de-scale
        return [p * self._sigma + self._mu for p in preds]

    def predict(self, h: int) -> List[float]:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return self._recursive_forecast(h, training_flag=False)

    def predict_interval(
        self, h: int, level: float = 0.95
    ) -> Tuple[List[float], List[float], List[float]]:
        """Monte-Carlo dropout intervals."""
        import numpy as np

        if self._model is None:
            raise RuntimeError("model is not fitted")
        samples = np.array(
            [self._recursive_forecast(h, training_flag=True) for _ in range(self.mc_samples)]
        )
        mean = samples.mean(axis=0)
        lo_q = (1 - level) / 2 * 100
        hi_q = (1 + level) / 2 * 100
        lower = np.percentile(samples, lo_q, axis=0)
        upper = np.percentile(samples, hi_q, axis=0)
        return list(map(float, mean)), list(map(float, lower)), list(map(float, upper))

    @property
    def params(self) -> dict:
        return {
            "lookback": self.lookback,
            "units": self.units,
            "dropout": self.dropout,
            "epochs": self.epochs,
        }
