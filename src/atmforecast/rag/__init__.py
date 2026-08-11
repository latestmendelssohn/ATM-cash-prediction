"""
Retrieval-Augmented-Generation analyst layer.

Mirrors the FinApp RAG sample's stack:
    * ChromaDB          -- vector store for forecast reports & PDF financials
    * Gemini            -- embeddings + chat completion
    * LangChain         -- retrieval + prompt orchestration
    * PyPDF             -- ingest external cash-management / audit PDFs

The pure-Python ``report_builder`` turns forecast artifacts into natural-language
documents; everything else is library-based and imported lazily.
"""

from .report_builder import (  # noqa: F401
    build_forecast_report,
    build_backtest_report,
    build_cash_plan_report,
)
