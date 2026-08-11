"""In-memory session store (session-based pipeline, mirrors the RAG sample).

Each session keeps its own chat history and remembers which ATMs have been
indexed into the vector store, so a conversation can build context turn by turn.
For a multi-replica deployment swap this for Redis; the interface is the same.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    history: List[Tuple[str, str]] = field(default_factory=list)  # (role, text)
    indexed_atms: set = field(default_factory=set)

    def add_turn(self, role: str, text: str) -> None:
        self.history.append((role, text))
        self.last_seen = time.time()


class SessionStore:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def create(self) -> Session:
        sid = uuid.uuid4().hex
        with self._lock:
            s = Session(session_id=sid)
            self._sessions[sid] = s
            return s

    def get(self, sid: str) -> Session | None:
        with self._lock:
            self._evict_expired()
            return self._sessions.get(sid)

    def get_or_create(self, sid: str | None) -> Session:
        if sid:
            s = self.get(sid)
            if s is not None:
                return s
        return self.create()

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._sessions.items() if now - v.last_seen > self._ttl]
        for k in expired:
            self._sessions.pop(k, None)
