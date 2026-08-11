r"""
RAG analyst layer  (ChromaDB + Gemini + LangChain).
===================================================

Turns the numerical forecast artifacts into short Markdown *reports*, embeds
them into ChromaDB with Gemini embeddings, and answers plain-English questions
by retrieving the most relevant reports and asking Gemini (grounded prompt).
External PDFs (e.g. an RBI cash-management circular) can also be ingested.

The report-building + prompt-assembly helpers are pure Python (unit tested);
the vector store and LLM are imported lazily so nothing network-related loads
until it is actually used.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

SYSTEM_PROMPT = (
    "You are an ATM cash-management analyst for an Indian bank. Answer strictly "
    "from the provided CONTEXT (forecasts, backtest results, cash plans). Quote "
    "concrete figures in INR lakh/crore and cite the ATM id. If the context does "
    "not contain the answer, say so plainly. Be concise and actionable."
)


def load_env() -> None:
    """Read .env if python-dotenv is installed, so GOOGLE_API_KEY is picked up.

    Without this the README's "copy .env.example to .env" step has no effect and
    the key has to be exported by hand.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent / ".env")


# ---------------------------------------------------------------------------
# 1. Report building (pure Python)
# ---------------------------------------------------------------------------

def inr(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e7:
        return f"Rs {x / 1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"Rs {x / 1e5:,.2f} L"
    return f"Rs {x:,.0f}"


def forecast_report(atm_id, model, start_date, point, lower=None, upper=None,
                    location="") -> Dict[str, object]:
    h = len(point)
    peak = max(range(h), key=lambda i: point[i])
    lines = [
        f"# Cash-demand forecast for {atm_id}" + (f" ({location})" if location else ""),
        f"Model: {model} | Horizon: {h} days from {start_date}.",
        f"- Total forecast demand: {inr(sum(point))}; average/day: {inr(sum(point)/h)}.",
        f"- Peak demand: day {peak+1} at {inr(point[peak])}.",
    ]
    if lower and upper:
        lines.append(f"- Day-1 95% interval: [{inr(lower[0])}, {inr(upper[0])}].")
    lines.append("Daily forecast: " + ", ".join(f"d{i+1}={inr(v)}" for i, v in enumerate(point)))
    return {"id": f"forecast::{atm_id}", "text": "\n".join(lines),
            "metadata": {"kind": "forecast", "atm_id": atm_id, "model": model}}


def backtest_report(atm_id, rows: List[dict]) -> Dict[str, object]:
    lines = [f"# Backtest leaderboard for {atm_id}",
             "Rolling-origin CV; lower is better; MASE<1 beats seasonal-naive.", ""]
    if rows:
        cols = list(rows[0])
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
        lines.append(f"\nBest model: {rows[0]['model']} (MASE {rows[0].get('MASE')}).")
    return {"id": f"backtest::{atm_id}", "text": "\n".join(lines),
            "metadata": {"kind": "backtest", "atm_id": atm_id,
                         "best_model": rows[0]["model"] if rows else ""}}


def cash_plan_report(atm_id, plan: Dict[str, object]) -> Dict[str, object]:
    lines = [
        f"# Cash-replenishment recommendation for {atm_id}",
        f"Service level {plan['service_level']:.0%} "
        f"(residual stock-out prob {plan['expected_stockout_prob']:.2%}).",
        f"- Load for the cycle: {inr(plan['cycle_load'])} "
        f"(safety stock {inr(plan['safety_stock'])}).",
    ]
    if "suggested_topup" in plan:
        lines.append(f"- Current balance {inr(plan['current_balance'])}; "
                     f"top-up now {inr(plan['suggested_topup'])}.")
    return {"id": f"cashplan::{atm_id}", "text": "\n".join(lines),
            "metadata": {"kind": "cash_plan", "atm_id": atm_id,
                         "service_level": plan["service_level"]}}


def build_prompt(question: str, contexts: List[Dict[str, object]]) -> str:
    if contexts:
        blocks = [f"[{i}] ({c.get('metadata', {}).get('kind', 'doc')})\n{c.get('text', '')}"
                  for i, c in enumerate(contexts, 1)]
        ctx = "\n\n".join(blocks)
    else:
        ctx = "(no relevant documents found)"
    return f"Use ONLY the context below.\n\nCONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nANSWER:"


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# 2. RAG analyst (ChromaDB + Gemini + LangChain) -- lazy imports
# ---------------------------------------------------------------------------

class Analyst:
    def __init__(self, chroma_dir="./chroma_db", collection="atm_reports",
                 model=None, embed_model=None, top_k=4, api_key=None):
        load_env()
        self.chroma_dir = chroma_dir
        self.collection_name = collection
        # Chat model: an alias, so this keeps working as Google retires versions.
        # Embeddings: pinned, because changing the model changes the vector
        # dimension and invalidates an existing index.
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        self.embed_model = embed_model or os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.top_k = top_k
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._col = self._emb = self._llm = None

    def _ensure(self):
        if self._col is not None:
            return
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY not set (copy .env.example to .env).")
        import chromadb
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=self.chroma_dir)
        self._col = client.get_or_create_collection(self.collection_name,
                                                    metadata={"hnsw:space": "cosine"})
        self._emb = GoogleGenerativeAIEmbeddings(model=self.embed_model, google_api_key=self.api_key)
        self._llm = ChatGoogleGenerativeAI(model=self.model, temperature=0.1,
                                           google_api_key=self.api_key)

    def add_reports(self, reports: List[Dict[str, object]]) -> int:
        if not reports:
            return 0
        self._ensure()
        self._col.upsert(
            ids=[str(r["id"]) for r in reports],
            documents=[str(r["text"]) for r in reports],
            metadatas=[dict(r.get("metadata", {})) for r in reports],
            embeddings=self._emb.embed_documents([str(r["text"]) for r in reports]),
        )
        return len(reports)

    def add_pdf(self, pdf_path: str) -> int:
        from pypdf import PdfReader

        text = "\n".join((p.extract_text() or "") for p in PdfReader(pdf_path).pages)
        name = Path(pdf_path).stem
        return self.add_reports([
            {"id": f"pdf::{name}::{i}", "text": c, "metadata": {"kind": "pdf", "source": name}}
            for i, c in enumerate(chunk_text(text))
        ])

    def _retrieve(self, question, atm_id=None):
        self._ensure()
        res = self._col.query(query_embeddings=[self._emb.embed_query(question)],
                              n_results=self.top_k,
                              where={"atm_id": atm_id} if atm_id else None)
        return [{"text": d, "metadata": m} for d, m in
                zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0])]

    def ask(self, question: str, atm_id: Optional[str] = None) -> Dict[str, object]:
        ctx = self._retrieve(question, atm_id)
        resp = self._llm.invoke([("system", SYSTEM_PROMPT),
                                 ("human", build_prompt(question, ctx))])
        return {"answer": getattr(resp, "content", str(resp)), "sources": ctx}

    def stream(self, question: str, atm_id: Optional[str] = None) -> Iterator[str]:
        ctx = self._retrieve(question, atm_id)
        for chunk in self._llm.stream([("system", SYSTEM_PROMPT),
                                       ("human", build_prompt(question, ctx))]):
            piece = getattr(chunk, "content", None)
            if piece:
                yield piece

    def count(self) -> int:
        self._ensure()
        return self._col.count()
