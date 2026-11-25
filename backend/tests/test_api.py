import pytest
from fastapi.testclient import TestClient

from conftest import FASTAPI_COURSE, RAG_COURSE, rate_limit_error, text_reply, tool_call


@pytest.fixture
def api(app_module, seeded_rag, monkeypatch):
    # No context manager: the startup hook (which ingests ../docs) never runs
    monkeypatch.setattr(app_module, "rag_system", seeded_rag)
    return TestClient(app_module.app)


def test_query_creates_session_and_returns_sources(api, fake_llm):
    fake_llm.script(tool_call("get_course_outline", {"course_name": "FastAPI"}), text_reply("Two lessons."))

    response = api.post("/api/query", json={"query": "Outline of the FastAPI course?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Two lessons."
    assert body["session_id"]
    assert body["sources"] == [{"text": FASTAPI_COURSE["title"], "url": FASTAPI_COURSE["link"]}]


def test_new_conversations_get_distinct_sessions(api, fake_llm):
    fake_llm.script(text_reply("a"), text_reply("b"))

    first = api.post("/api/query", json={"query": "hi"}).json()["session_id"]
    second = api.post("/api/query", json={"query": "hi"}).json()["session_id"]

    assert first != second


def test_existing_session_is_continued(api, fake_llm):
    fake_llm.script(text_reply("first answer"), text_reply("second answer"))

    session = api.post("/api/query", json={"query": "first question"}).json()["session_id"]
    response = api.post("/api/query", json={"query": "second", "session_id": session})

    assert response.json()["session_id"] == session
    assert "first question" in fake_llm.calls[1]["system"]


def test_missing_query_is_rejected(api):
    assert api.post("/api/query", json={"session_id": "x"}).status_code == 422


def test_llm_failure_returns_500_with_detail(api, fake_llm):
    fake_llm.script(rate_limit_error())

    response = api.post("/api/query", json={"query": "hi"})

    assert response.status_code == 500
    assert "rate limited" in response.json()["detail"]


def test_course_stats(api):
    body = api.get("/api/courses").json()

    assert body["total_courses"] == 2
    assert sorted(body["course_titles"]) == sorted([RAG_COURSE["title"], FASTAPI_COURSE["title"]])


def test_delete_session_forgets_history(api, fake_llm):
    fake_llm.script(text_reply("first answer"), text_reply("fresh"))
    session = api.post("/api/query", json={"query": "secret question"}).json()["session_id"]

    assert api.delete(f"/api/session/{session}").json() == {"status": "ok"}
    api.post("/api/query", json={"query": "again", "session_id": session})

    assert "secret question" not in fake_llm.calls[1]["system"]


def test_delete_session_is_idempotent(api):
    assert api.delete("/api/session/never-existed").status_code == 200
    assert api.delete("/api/session/never-existed").status_code == 200


def test_frontend_is_served(api):
    response = api.get("/")

    assert response.status_code == 200
    assert 'id="chatInput"' in response.text
