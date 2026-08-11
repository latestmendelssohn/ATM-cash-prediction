r"""
Gemini chat wrapper with streaming.
===================================

Thin adapter around ``langchain-google-genai``'s ``ChatGoogleGenerativeAI`` that
exposes both a blocking ``complete`` and a token-level ``stream`` generator used
by the FastAPI SSE endpoint. Keeping the LLM behind this interface means the
rest of the code never imports the SDK directly and can be mocked in tests.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

from ..config import get_settings

SYSTEM_PROMPT = (
    "You are an ATM cash-management analyst for an Indian bank. You answer "
    "strictly from the provided CONTEXT, which contains time-series forecasts, "
    "backtest results and cash-replenishment recommendations. Quote concrete "
    "figures (in INR lakh/crore) and cite the ATM id. If the context does not "
    "contain the answer, say so plainly rather than guessing. Be concise and "
    "operationally actionable."
)


class GeminiLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> None:
        s = get_settings()
        self.model = model or s.gemini_model
        self.temperature = 0.1 if temperature is None else temperature
        self.api_key = api_key or s.google_api_key
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._client = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
            google_api_key=self.api_key,
        )

    def _messages(self, prompt: str, system: Optional[str] = None) -> List[tuple]:
        return [("system", system or SYSTEM_PROMPT), ("human", prompt)]

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        self._ensure()
        resp = self._client.invoke(self._messages(prompt, system))
        return resp.content if hasattr(resp, "content") else str(resp)

    def stream(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        """Yield response text chunk-by-chunk for server-sent events."""
        self._ensure()
        for chunk in self._client.stream(self._messages(prompt, system)):
            piece = getattr(chunk, "content", None)
            if piece:
                yield piece
