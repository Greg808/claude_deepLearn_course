"""Browser tests: real frontend + real backend, only Claude is faked."""
import socket
import threading
import time

import pytest
import uvicorn
from playwright.sync_api import Page, expect

from conftest import RAG_COURSE, rate_limit_error, text_reply, tool_call

pytestmark = pytest.mark.e2e

# Offline stand-in for the CDN copy of marked: escapes instead of rendering markdown
MARKED_STUB = "window.marked={parse:t=>{const d=document.createElement('div');d.textContent=t;return '<p>'+d.innerHTML+'</p>';}};"


@pytest.fixture(scope="module")
def server_url(app_module):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    # lifespan off: the startup hook would ingest ../docs
    server = uvicorn.Server(uvicorn.Config(app_module.app, port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join()


@pytest.fixture
def chat(page: Page, server_url, app_module, seeded_rag, monkeypatch):
    monkeypatch.setattr(app_module, "rag_system", seeded_rag)

    def route(r):
        if r.request.url.startswith(server_url):
            r.continue_()
        elif r.request.url.endswith("marked.min.js"):
            r.fulfill(content_type="application/javascript", body=MARKED_STUB)
        else:
            r.abort()

    page.route("**/*", route)
    page.goto(server_url)
    return page


def messages(page):
    return page.locator("#chatMessages .message")


def test_course_stats_load(chat):
    expect(chat.locator("#totalCourses")).to_have_text("2")
    expect(chat.locator("#courseTitles .course-title-item")).to_have_count(2)


def test_question_shows_answer_with_linked_source(chat, fake_llm):
    fake_llm.script(tool_call("get_course_outline", {"course_name": "Retrieval Augmented Generation"}), text_reply("It has three lessons."))

    chat.fill("#chatInput", "Outline of the RAG course?")
    chat.click("#sendButton")

    expect(messages(chat).nth(1)).to_have_text("Outline of the RAG course?")
    answer = messages(chat).nth(2)
    expect(answer.locator(".message-content")).to_have_text("It has three lessons.")
    expect(answer.locator(".sources-header")).to_have_text("Sources (1)")
    link = answer.locator("a.source-item")
    expect(link).to_have_attribute("href", RAG_COURSE["link"])
    expect(link).to_have_attribute("target", "_blank")
    expect(chat.locator("#chatInput")).to_be_enabled()


def test_sources_are_deduplicated_and_escaped(chat, seeded_rag, monkeypatch):
    sources = [
        {"text": "Course A - Lesson 1", "url": "https://example.com/a/1"},
        {"text": "Course A - Lesson 1", "url": "https://example.com/a/1"},
        {"text": "No link source", "url": None},
        {"text": "<img src=x onerror=alert(1)>", "url": None},
    ]
    monkeypatch.setattr(seeded_rag, "query", lambda query, session_id=None: ("Answer.", sources))

    chat.fill("#chatInput", "anything")
    chat.press("#chatInput", "Enter")

    answer = messages(chat).nth(2)
    expect(answer.locator(".sources-header")).to_have_text("Sources (3)")
    expect(answer.locator("a.source-item")).to_have_count(1)
    expect(answer.locator("span.source-item")).to_have_text(["No link source", "<img src=x onerror=alert(1)>"])
    expect(answer.locator("img")).to_have_count(0)


def test_suggested_question_is_sent(chat, fake_llm):
    fake_llm.script(text_reply("Yes."))
    chat.click(".suggested-header")
    button = chat.locator(".suggested-item").first
    question = button.get_attribute("data-question")

    button.click()

    expect(messages(chat).nth(1)).to_have_text(question)
    expect(messages(chat).nth(2)).to_have_text("Yes.")
    assert question in fake_llm.calls[0]["messages"][0]["content"]


def test_server_error_is_shown_and_input_reenabled(chat, fake_llm):
    fake_llm.script(rate_limit_error())

    chat.fill("#chatInput", "hi")
    chat.click("#sendButton")

    expect(messages(chat).nth(2)).to_have_text("Error: Query failed")
    expect(chat.locator("#chatInput")).to_be_enabled()


def test_new_chat_deletes_session_and_starts_fresh(chat, fake_llm, seeded_rag):
    fake_llm.script(text_reply("first"), text_reply("second"))
    chat.fill("#chatInput", "first question")
    chat.click("#sendButton")
    expect(messages(chat)).to_have_count(3)
    (old_session,) = seeded_rag.session_manager.sessions

    with chat.expect_request(lambda r: r.method == "DELETE") as deleted:
        chat.click("#newChatButton")
    assert deleted.value.url.endswith(f"/api/session/{old_session}")
    expect(messages(chat)).to_have_count(1)
    expect(chat.locator(".welcome-message")).to_be_visible()

    with chat.expect_request("**/api/query") as query:
        chat.fill("#chatInput", "second question")
        chat.click("#sendButton")
    assert query.value.post_data_json["session_id"] is None
    expect(messages(chat)).to_have_count(3)
    assert old_session not in seeded_rag.session_manager.sessions
    assert "first question" not in fake_llm.calls[1]["system"]


def test_new_chat_discards_answer_still_in_flight(chat):
    held = []
    chat.route("**/api/query", lambda r: held.append(r))

    chat.fill("#chatInput", "slow question")
    chat.click("#sendButton")
    expect(chat.locator(".loading")).to_be_visible()
    chat.click("#newChatButton")

    expect(messages(chat)).to_have_count(1)
    expect(chat.locator("#chatInput")).to_be_enabled()
    assert len(held) == 1


def test_theme_toggle_switches_and_persists(chat):
    html = chat.locator("html")
    toggle = chat.locator("#themeToggle")
    expect(html).to_have_attribute("data-theme", "dark")
    expect(toggle).to_have_attribute("aria-label", "Switch to light theme")

    toggle.click()
    expect(html).to_have_attribute("data-theme", "light")
    expect(toggle).to_have_attribute("aria-label", "Switch to dark theme")
    bg = chat.evaluate("getComputedStyle(document.body).getPropertyValue('--background').trim()")
    assert bg == "#f1f5f9"

    chat.reload()
    expect(html).to_have_attribute("data-theme", "light")

    toggle.focus()
    chat.keyboard.press("Enter")
    expect(html).to_have_attribute("data-theme", "dark")
