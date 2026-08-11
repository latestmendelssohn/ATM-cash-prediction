"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    atm_id: str = Field(..., examples=["ATM001"])
    model: str = Field("holt_winters", description="mean|drift|moving_average|seasonal_naive|holt_winters|sarima|prophet|lstm")
    horizon: int = Field(14, ge=1, le=90)
    target: str = "net_cash_out"
    level: float = Field(0.95, gt=0, lt=1)


class ForecastResponse(BaseModel):
    atm_id: str
    model: str
    horizon: int
    forecast_start: str
    dates: List[str]
    point: List[float]
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None
    level: float
    residual_std: Optional[float] = None


class CashPlanRequest(BaseModel):
    atm_id: str
    model: str = "holt_winters"
    horizon: int = Field(14, ge=1, le=90)
    service_level: float = Field(0.95, gt=0, lt=1)
    current_balance: Optional[float] = None


class IngestResponse(BaseModel):
    session_id: str
    documents_added: int
    total_documents: int


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    atm_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[dict]


class SessionResponse(BaseModel):
    session_id: str
