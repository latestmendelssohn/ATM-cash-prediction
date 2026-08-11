"""
atmforecast
===========

Time-series forecasting of daily ATM cash demand and end-of-day cash balance,
with a Gemini-powered Retrieval-Augmented-Generation (RAG) analyst layer.

The package is organised into loosely-coupled sub-packages:

    data/        synthetic data generation + loading/aggregation
    features/    calendar & holiday feature engineering
    models/      forecasting models (baselines, Holt-Winters, SARIMA, Prophet, LSTM)
    evaluation/  error metrics + rolling-origin backtesting
    operations/  cash-inventory optimisation on top of the forecast
    rag/         ChromaDB + Gemini + LangChain natural-language analyst
    api/         FastAPI streaming service

Design note
-----------
The `evaluation.metrics`, `models.baselines`, `models.holt_winters` and
`data.generate_synthetic` modules are written in **pure Python (standard
library only)** so the mathematical core is transparent and dependency-free.
The heavier modules (SARIMA, Prophet, LSTM, RAG) build on the scientific /
LLM stack declared in ``requirements.txt``.
"""

__version__ = "0.1.0"
