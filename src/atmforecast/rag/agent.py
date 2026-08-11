r"""
RAG analyst agent  (LangChain orchestration).
=============================================

Ties the retriever (ChromaDB) to the generator (Gemini). Given a natural-
language question about ATM cash, it:

    1. retrieves the top-k most relevant forecast / backtest / cash-plan chunks
       (optionally filtered to a specific ATM),
    2. assembles a grounded prompt with an explicit CONTEXT block, and
    3. asks Gemini to answer -- streaming or blocking.

The prompt-assembly step is pure Python (``build_prompt``) and unit tested; the
network calls are delegated to :class:`GeminiLLM` and :class:`ForecastVectorStore`.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from .llm import GeminiLLM
from .vectorstore import ForecastVectorStore


def build_prompt(question: str, contexts: List[Dict[str, object]]) -> str:
    """Assemble the grounded RAG prompt from retrieved chunks (pure stdlib)."""
    if contexts:
        blocks = []
        for i, c in enumerate(contexts, start=1):
            meta = c.get("metadata", {}) or {}
            tag = meta.get("kind", "doc")
            atm = meta.get("atm_id", "")
            header = f"[{i}] ({tag}{' - ' + atm if atm else ''})"
            blocks.append(f"{header}\n{c.get('text', '')}")
        context_str = "\n\n".join(blocks)
    else:
        context_str = "(no relevant documents found)"

    return (
        "Answer the question using ONLY the context below.\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


class RAGAnalyst:
    def __init__(
        self,
        store: Optional[ForecastVectorStore] = None,
        llm: Optional[GeminiLLM] = None,
        top_k: int = 4,
    ) -> None:
        self.store = store or ForecastVectorStore()
        self.llm = llm or GeminiLLM()
        self.top_k = top_k

    def _retrieve(self, question: str, atm_id: Optional[str]) -> List[Dict[str, object]]:
        where = {"atm_id": atm_id} if atm_id else None
        return self.store.query(question, top_k=self.top_k, where=where)

    def answer(self, question: str, atm_id: Optional[str] = None) -> Dict[str, object]:
        """Blocking answer with the retrieved sources attached."""
        contexts = self._retrieve(question, atm_id)
        prompt = build_prompt(question, contexts)
        text = self.llm.complete(prompt)
        return {"answer": text, "sources": contexts}

    def stream_answer(
        self, question: str, atm_id: Optional[str] = None
    ) -> Iterator[str]:
        """Token-streaming answer (used by the SSE endpoint)."""
        contexts = self._retrieve(question, atm_id)
        prompt = build_prompt(question, contexts)
        yield from self.llm.stream(prompt)
