r"""
ChromaDB vector store with Gemini embeddings.
=============================================

Persists forecast reports (and ingested PDF financial documents) as embedded
chunks so the analyst can retrieve the most relevant context for a question.

Embeddings use Google's ``text-embedding-004`` via LangChain's
``GoogleGenerativeAIEmbeddings``. Retrieval is cosine-similarity top-k, with
optional metadata filtering (e.g. restrict to one ATM or one ``kind``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..config import get_settings


class ForecastVectorStore:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection: str = "atm_forecast_reports",
        embedding_model: Optional[str] = None,
    ) -> None:
        s = get_settings()
        self.persist_dir = persist_dir or s.chroma_dir
        self.collection_name = collection
        self.embedding_model = embedding_model or s.embedding_model
        self._client = None
        self._collection = None
        self._embeddings = None

    # ------------------------------------------------------------- lazy init
    def _ensure(self):
        if self._collection is not None:
            return
        import chromadb
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure()
        return self._embeddings.embed_documents(texts)

    # ---------------------------------------------------------------- ingest
    def add_reports(self, reports: List[Dict[str, object]]) -> int:
        """Add a batch of ``{id, text, metadata}`` report documents."""
        if not reports:
            return 0
        self._ensure()
        ids = [str(r["id"]) for r in reports]
        docs = [str(r["text"]) for r in reports]
        metas = [dict(r.get("metadata", {})) for r in reports]  # type: ignore[arg-type]
        embeddings = self._embed(docs)
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        return len(ids)

    def add_pdf(self, pdf_path: str, chunk_chars: int = 1200, overlap: int = 150) -> int:
        """Ingest an external PDF (e.g. RBI cash-management circular) with PyPDF."""
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        full = "\n".join((page.extract_text() or "") for page in reader.pages)
        chunks = _chunk_text(full, chunk_chars, overlap)
        name = Path(pdf_path).stem
        reports = [
            {
                "id": f"pdf::{name}::{i}",
                "text": c,
                "metadata": {"kind": "pdf", "source": name, "chunk": i},
            }
            for i, c in enumerate(chunks)
        ]
        return self.add_reports(reports)

    # ---------------------------------------------------------------- query
    def query(
        self, question: str, top_k: int = 4, where: Optional[Dict] = None
    ) -> List[Dict[str, object]]:
        """Return the top-k most similar chunks to ``question``."""
        self._ensure()
        q_emb = self._embeddings.embed_query(question)
        res = self._collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where or None,
        )
        out: List[Dict[str, object]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"text": doc, "metadata": meta, "distance": dist})
        return out

    def count(self) -> int:
        self._ensure()
        return self._collection.count()

    def reset(self) -> None:
        self._ensure()
        self._client.delete_collection(self.collection_name)
        self._collection = None


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Simple character-window chunker with overlap (stdlib)."""
    text = text.strip()
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks
