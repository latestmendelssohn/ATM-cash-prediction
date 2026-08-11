"""Tests for the in-memory session store (pure Python, no FastAPI)."""
import time

from atmforecast.api.session import SessionStore


def test_create_returns_unique_ids():
    store = SessionStore()
    a = store.create()
    b = store.create()
    assert a.session_id != b.session_id
    assert store.get(a.session_id) is a


def test_get_or_create_reuses_existing():
    store = SessionStore()
    s = store.create()
    same = store.get_or_create(s.session_id)
    assert same.session_id == s.session_id


def test_get_or_create_makes_new_when_unknown():
    store = SessionStore()
    s = store.get_or_create("does-not-exist")
    assert store.get(s.session_id) is s


def test_history_accumulates():
    store = SessionStore()
    s = store.create()
    s.add_turn("user", "hi")
    s.add_turn("assistant", "hello")
    assert [r for r, _ in s.history] == ["user", "assistant"]


def test_expired_sessions_are_evicted():
    store = SessionStore(ttl_seconds=0)
    s = store.create()
    time.sleep(0.01)
    assert store.get(s.session_id) is None
