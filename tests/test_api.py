"""API smoke tests.

These only run when FastAPI is installed. That matters: the bug they guard against
(the app failing to build at import time) is invisible unless FastAPI is present,
which is why CI runs the suite a second time with it installed.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import app as A  # noqa: E402


@pytest.fixture(scope="module")
def client():
    assert A.app is not None, "create_app() returned None, so the API never built"
    return TestClient(A.app)


def test_app_builds_when_fastapi_is_installed(client):
    """Importing app.py must not raise once FastAPI is installed."""
    assert sorted(client.app.openapi()["paths"]) == [
        "/cash-plan", "/chat", "/chat/stream", "/forecast",
        "/health", "/index/{atm_id}", "/ingest/pdf",
    ]


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_forecast_returns_the_requested_horizon(client):
    body = client.post("/forecast", json={"atm_id": "ATM1", "horizon": 3}).json()
    assert len(body["point"]) == 3 and len(body["lower"]) == 3


def test_unknown_atm_is_a_400(client):
    assert client.post("/forecast", json={"atm_id": "NOPE"}).status_code == 400


def test_cash_plan_costs_set_the_service_level(client):
    body = client.post("/cash-plan",
                       json={"atm_id": "ATM1", "horizon": 14, "cu": 9, "co": 1}).json()
    assert body["service_level"] == pytest.approx(0.9)


def test_pdf_upload_does_not_trust_the_client_filename(client, tmp_path):
    """A "../" filename must not be joined onto a path."""
    r = client.post("/ingest/pdf",
                    files={"file": ("../../evil.pdf", b"%PDF-1.4 fake", "application/pdf")})
    # No Gemini key in CI, so ingestion fails; the point is that it fails cleanly
    # instead of writing outside the temp directory.
    assert r.status_code in (400, 503)
    assert not (tmp_path.parent / "evil.pdf").exists()
