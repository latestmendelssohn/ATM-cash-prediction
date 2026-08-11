"""
Configuration loader.

Merges three sources (in increasing priority):
    1. ``config/config.yaml``        -- pipeline defaults (checked into git)
    2. environment variables         -- deployment overrides
    3. a local ``.env`` file         -- developer secrets (never committed)

Only the library-based modules (SARIMA, RAG, API) import this. The pure-Python
core deliberately has no dependency on it.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # deferred import so the pure-Python core stays dependency-free

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Settings:
    """Lightweight settings object exposing YAML config + secret env vars."""

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        # Best-effort .env loading (no hard dependency on python-dotenv).
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except Exception:  # pragma: no cover
            pass

        self.raw: Dict[str, Any] = _load_yaml(Path(config_path))

        # Secrets / deployment knobs from the environment.
        self.google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
        self.gemini_model: str = os.getenv(
            "GEMINI_MODEL", self.raw.get("rag", {}).get("model", "gemini-1.5-flash")
        )
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL",
            self.raw.get("rag", {}).get("embedding_model", "models/text-embedding-004"),
        )
        self.chroma_dir: str = os.getenv(
            "CHROMA_DIR", self.raw.get("rag", {}).get("chroma_dir", "./chroma_db")
        )
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.stream_chunk_tokens: int = int(os.getenv("STREAM_CHUNK_TOKENS", "8"))

    # convenience accessors -------------------------------------------------
    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def forecast(self) -> Dict[str, Any]:
        return self.raw.get("forecast", {})

    @property
    def backtest(self) -> Dict[str, Any]:
        return self.raw.get("backtest", {})

    @property
    def models(self) -> Dict[str, Any]:
        return self.raw.get("models", {})

    @property
    def operations(self) -> Dict[str, Any]:
        return self.raw.get("operations", {})

    @property
    def rag(self) -> Dict[str, Any]:
        return self.raw.get("rag", {})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
