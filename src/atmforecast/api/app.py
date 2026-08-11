r"""
FastAPI application.
====================

Endpoints
---------
GET  /health                 liveness probe
POST /session                open a new analyst session
POST /forecast               forecast an ATM's cash demand (+ prediction band)
POST /cash-plan              recommended cash load for a service level
POST /index/{atm_id}         build & embed the RAG reports for an ATM
POST /ingest/pdf             ingest an external PDF (RBI circular, audit, ...)
POST /chat                   blocking RAG answer
POST /chat/stream            token-streaming RAG answer (Server-Sent Events)

Design notes
------------
* Session-based: every chat carries a ``session_id`` (created on demand) so a
  conversation accumulates context. Mirrors the FinApp RAG sample.
* Streaming: ``/chat/stream`` returns ``text/event-stream`` so the browser can
  render tokens as they arrive.
* Error handling: data / model / LLM errors are converted to clean HTTP 4xx/5xx
  responses instead of leaking tracebacks.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from ..config import get_settings
from .schemas import (
    CashPlanRequest,
    ChatRequest,
    ChatResponse,
    ForecastRequest,
    ForecastResponse,
    IngestResponse,
    SessionResponse,
)
from .session import SessionStore


def create_app():
    """Application factory (keeps import side-effects out of module import)."""
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    from .. import service

    settings = get_settings()
    data_path = settings.data.get("raw_path", "data/raw/atm_transactions.csv")

    app = FastAPI(
        title="ATM Cash Forecasting & RAG Analyst",
        version="0.1.0",
        description="Time-series cash-demand forecasting with a Gemini-powered analyst.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sessions = SessionStore()
    # RAGAnalyst is created lazily so the API boots even without an API key.
    _analyst_holder: dict = {}

    def get_analyst():
        if "analyst" not in _analyst_holder:
            from ..rag.agent import RAGAnalyst

            _analyst_holder["analyst"] = RAGAnalyst(top_k=settings.rag.get("top_k", 4))
        return _analyst_holder["analyst"]

    # ---------------------------------------------------------------- health
    @app.get("/health")
    def health():
        return {"status": "ok", "has_api_key": bool(settings.google_api_key)}

    @app.post("/session", response_model=SessionResponse)
    def open_session():
        return SessionResponse(session_id=sessions.create().session_id)

    # -------------------------------------------------------------- forecast
    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(req: ForecastRequest):
        try:
            fc = service.forecast_atm(
                data_path, req.atm_id, model=req.model,
                horizon=req.horizon, target=req.target, level=req.level,
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ImportError as e:
            raise HTTPException(status_code=501, detail=f"model backend unavailable: {e}")
        return ForecastResponse(**{k: fc[k] for k in ForecastResponse.model_fields})

    @app.post("/cash-plan")
    def cash_plan(req: CashPlanRequest):
        try:
            fc = service.forecast_atm(
                data_path, req.atm_id, model=req.model, horizon=req.horizon
            )
            plan = service.cash_plan_atm(
                fc, service_level=req.service_level, current_balance=req.current_balance
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return plan

    # ----------------------------------------------------------- RAG indexing
    @app.post("/index/{atm_id}", response_model=IngestResponse)
    def index_atm(atm_id: str, session_id: Optional[str] = None,
                  model: str = "holt_winters", horizon: int = 14):
        sess = sessions.get_or_create(session_id)
        try:
            docs = service.build_reports_for_atm(data_path, atm_id, model=model, horizon=horizon)
            analyst = get_analyst()
            added = analyst.store.add_reports(docs)
            total = analyst.store.count()
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:  # e.g. missing API key for embeddings
            raise HTTPException(status_code=503, detail=str(e))
        sess.indexed_atms.add(atm_id)
        return IngestResponse(session_id=sess.session_id, documents_added=added, total_documents=total)

    @app.post("/ingest/pdf", response_model=IngestResponse)
    async def ingest_pdf(file: UploadFile = File(...), session_id: Optional[str] = None):
        sess = sessions.get_or_create(session_id)
        tmp = f"/tmp/{file.filename}"
        try:
            with open(tmp, "wb") as fh:
                fh.write(await file.read())
            analyst = get_analyst()
            added = analyst.store.add_pdf(tmp)
            total = analyst.store.count()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to ingest PDF: {e}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return IngestResponse(session_id=sess.session_id, documents_added=added, total_documents=total)

    # -------------------------------------------------------------- chat
    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        sess = sessions.get_or_create(req.session_id)
        try:
            result = get_analyst().answer(req.question, atm_id=req.atm_id)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        sess.add_turn("user", req.question)
        sess.add_turn("assistant", result["answer"])
        return ChatResponse(session_id=sess.session_id, answer=result["answer"],
                            sources=result.get("sources", []))

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest):
        sess = sessions.get_or_create(req.session_id)

        def event_gen():
            # first event carries the session id
            yield f"event: session\ndata: {sess.session_id}\n\n"
            collected = []
            try:
                for token in get_analyst().stream_answer(req.question, atm_id=req.atm_id):
                    collected.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
            except RuntimeError as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                return
            sess.add_turn("user", req.question)
            sess.add_turn("assistant", "".join(collected))
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return app


# ASGI entrypoint: `uvicorn atmforecast.api.app:app`
app = create_app()
