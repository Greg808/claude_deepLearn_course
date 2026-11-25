"""Shared fixtures: offline setup, a scripted fake Claude client, temp course data."""
import dataclasses
import os
from pathlib import Path

# Must run before any backend import: config.py calls load_dotenv() (which never
# overrides variables already set), and the embedding model must come from the
# local cache only.
BACKEND_DIR = Path(__file__).resolve().parents[1]
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import anthropic
import httpx
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from config import config
from rag_system import RAGSystem
from vector_store import VectorStore


# --- Fake Claude -------------------------------------------------------------

def text_reply(text: str) -> Message:
    return _message([TextBlock(type="text", text=text)], "end_turn")


def tool_call(name: str, tool_input: dict, tool_id: str = "toolu_1") -> Message:
    return _message([ToolUseBlock(type="tool_use", id=tool_id, name=name, input=tool_input)], "tool_use")


def multi_tool_call(*calls: tuple) -> Message:
    """calls: (name, input, id) tuples, all in one assistant turn"""
    blocks = [ToolUseBlock(type="tool_use", id=i, name=n, input=inp) for n, inp, i in calls]
    return _message(blocks, "tool_use")


def _message(content, stop_reason) -> Message:
    return Message(
        id="msg_test", type="message", role="assistant", model="test-model",
        content=content, stop_reason=stop_reason, stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: replays scripted replies, records every call."""

    def __init__(self):
        self.messages = self
        self.calls = []
        self._script = []

    def script(self, *replies):
        """Queue Message objects (returned) or exceptions (raised), one per API call."""
        self._script.extend(replies)

    def create(self, **params):
        self.calls.append({**params, "messages": list(params["messages"])})
        if not self._script:
            raise AssertionError(f"Unscripted Claude call #{len(self.calls)}")
        reply = self._script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def fake_llm():
    return FakeAnthropic()


# --- Course fixtures ---------------------------------------------------------

RAG_COURSE = {
    "title": "Introduction to Retrieval Augmented Generation",
    "link": "https://example.com/courses/rag",
    "instructor": "Ada Lovelace",
    "lessons": [
        (0, "Welcome", "https://example.com/courses/rag/lesson/0",
         "Welcome to the course. Retrieval augmented generation combines search with language models."),
        (1, "Embeddings and Vector Search", "https://example.com/courses/rag/lesson/1",
         "Embeddings turn text into vectors. Similar sentences end up close together in vector space."),
        (2, "Chunking Strategies", None,
         "Long documents are split into chunks. Overlapping chunks keep context across boundaries."),
    ],
}

FASTAPI_COURSE = {
    "title": "Building Web Servers with FastAPI",
    "link": "https://example.com/courses/fastapi",
    "instructor": "Grace Hopper",
    "lessons": [
        (1, "Routing Basics", "https://example.com/courses/fastapi/lesson/1",
         "A route maps an HTTP path to a Python function. Decorators register each route."),
        (3, "Dependency Injection", "https://example.com/courses/fastapi/lesson/3",
         "Dependencies are declared as function parameters. The framework resolves them per request."),
    ],
}


def course_text(course: dict) -> str:
    lines = [f"Course Title: {course['title']}"]
    if course["link"]:
        lines.append(f"Course Link: {course['link']}")
    if course["instructor"]:
        lines.append(f"Course Instructor: {course['instructor']}")
    lines.append("")
    for number, title, link, content in course["lessons"]:
        lines.append(f"Lesson {number}: {title}")
        if link:
            lines.append(f"Lesson Link: {link}")
        lines.append(content)
    return "\n".join(lines) + "\n"


@pytest.fixture
def course_dir(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "rag.txt").write_text(course_text(RAG_COURSE), encoding="utf-8")
    (folder / "fastapi.txt").write_text(course_text(FASTAPI_COURSE), encoding="utf-8")
    return folder


@pytest.fixture
def test_config(tmp_path):
    return dataclasses.replace(config, CHROMA_PATH=str(tmp_path / "chroma"))


@pytest.fixture
def store(test_config):
    return VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)


@pytest.fixture
def rag(test_config, fake_llm):
    system = RAGSystem(test_config)
    system.ai_generator.client = fake_llm
    return system


@pytest.fixture
def seeded_rag(rag, course_dir):
    assert rag.add_course_folder(str(course_dir))[0] == 2
    return rag


@pytest.fixture
def seeded_store(seeded_rag):
    return seeded_rag.vector_store


# --- App ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def app_module(tmp_path_factory):
    """Import app.py lazily, with its import-time RAGSystem pointed at a temp store."""
    # app.py resolves ../frontend from the cwd on every request, so this stays for
    # the session. Not done at conftest import: pytest would then miss testpaths.
    os.chdir(BACKEND_DIR)
    config.CHROMA_PATH = str(tmp_path_factory.mktemp("app_chroma"))
    import app
    return app
