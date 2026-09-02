import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.datasets import ROOT, build

QUESTION = "What share of finance jobs have a DC pension?"


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    path = tmp_path / "ashe.db"
    build(ROOT / "sources/fixture.json", path)
    monkeypatch.setenv("ASHE_DB_PATH", str(path))
    return path


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture
def configured(snapshot, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-not-a-key")


def test_ask_returns_answer_sources_trace_and_verdict(configured, client, monkeypatch):
    seen = []

    def fake_agent(question):
        seen.append(question)
        return {
            "question": question,
            "answer": "In 2024, 65% of finance jobs had a DC pension.",
            "sources": ["P2 (2024)"],
            "tool_calls": [{"tool": "lookup_industry_membership_rate", "ok": True}],
            "critic": {"grounded": True, "advice_free": True, "clear_for_hr_reader": True, "issues": []},
        }

    monkeypatch.setattr(server, "ask_agent", fake_agent)
    response = client.post("/api/ask", json={"question": f"  {QUESTION}  "})

    assert response.status_code == 200
    assert seen == [QUESTION]
    body = response.json()
    assert body["sources"] == ["P2 (2024)"]
    assert body["tool_calls"][0]["tool"] == "lookup_industry_membership_rate"
    assert body["critic"]["grounded"] is True


@pytest.mark.parametrize("question", ["", "abc", "     ", "x" * 1001])
def test_ask_validates_question_length(configured, client, question):
    assert client.post("/api/ask", json={"question": question}).status_code in {400, 422}


def test_agent_failures_hide_internal_errors(configured, client, monkeypatch):
    def broken(_question):
        raise RuntimeError("All candidate models failed: {'gpt': 'secret detail'}")

    monkeypatch.setattr(server, "ask_agent", broken)
    response = client.post("/api/ask", json={"question": QUESTION})
    assert response.status_code == 500
    assert "secret detail" not in response.text


def test_ready_lists_fixture_coverage(snapshot, client):
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert body["tables"] == ["P12", "P2"]
    assert body["years"] == [2023, 2024]


def test_admin_endpoints_reject_wrong_or_unset_tokens(snapshot, client, monkeypatch):
    monkeypatch.delenv("ASHE_REFRESH_TOKEN", raising=False)
    assert client.get("/admin/source-status", headers={"Authorization": "Bearer "}).status_code == 401
    monkeypatch.setenv("ASHE_REFRESH_TOKEN", "local-test-refresh-secret")
    assert client.post("/admin/refresh", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/admin/refresh", headers={"Authorization": "local-test-refresh-secret"}).status_code == 401


def test_cors_allows_only_configured_origins(client):
    allowed = client.options("/api/ask", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    denied = client.options("/api/ask", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in denied.headers
