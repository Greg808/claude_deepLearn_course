import dataclasses

import anthropic
import pytest

from ai_generator import AIGenerator
from conftest import RAG_COURSE, connection_error, multi_tool_call, text_reply, tool_call, _message
from anthropic.types import TextBlock, ToolUseBlock
from rag_system import RAGSystem
from search_tools import Tool, ToolManager

AUTO = {"type": "auto"}
NONE = {"type": "none"}


class RecordingTool(Tool):
    def __init__(self, name="lookup"):
        self.name = name
        self.calls = []

    def get_tool_definition(self):
        return {"name": self.name, "description": "test tool", "input_schema": {"type": "object", "properties": {}}}

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return f"{self.name} result"


class RaisingTool(RecordingTool):
    def execute(self, **kwargs):
        raise RuntimeError("db down")


def make_manager(*tools):
    manager = ToolManager()
    for tool in tools:
        manager.register_tool(tool)
    return manager


@pytest.fixture
def generator(fake_llm):
    gen = AIGenerator(api_key="test-key-not-used", model="test-model")
    gen.client = fake_llm
    return gen


def ask(generator, manager, query="q", **kwargs):
    return generator.generate_response(query, tools=manager.get_tool_definitions(), tool_manager=manager, **kwargs)


def test_direct_answer_makes_one_call_with_tools_offered(generator, fake_llm, rag):
    fake_llm.script(text_reply("Python is a language."))
    tools = rag.tool_manager.get_tool_definitions()

    answer = generator.generate_response("What is Python?", tools=tools, tool_manager=rag.tool_manager)

    assert answer == "Python is a language."
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["tools"] == tools
    assert fake_llm.calls[0]["messages"] == [{"role": "user", "content": "What is Python?"}]


def test_text_blocks_are_joined(generator, fake_llm):
    fake_llm.script(_message([TextBlock(type="text", text="Part one. "), TextBlock(type="text", text="Part two.")], "end_turn"))

    assert generator.generate_response("q") == "Part one. Part two."


def test_tool_round_feeds_real_tool_output_back(generator, fake_llm, seeded_rag):
    manager = seeded_rag.tool_manager
    fake_llm.script(tool_call("get_course_outline", {"course_name": "Retrieval"}, "toolu_42"), text_reply("Here is the outline."))

    answer = ask(generator, manager, "Outline?")

    assert answer == "Here is the outline."
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[1]["tool_choice"] == AUTO  # a second round is still possible
    follow_up = fake_llm.calls[1]["messages"]
    assert [m["role"] for m in follow_up] == ["user", "assistant", "user"]
    (result,) = follow_up[2]["content"]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_42"
    for _, title, _, _ in RAG_COURSE["lessons"]:
        assert title in result["content"]


def test_parallel_tool_calls_return_in_one_message(generator, fake_llm, seeded_rag):
    manager = seeded_rag.tool_manager
    fake_llm.script(
        multi_tool_call(("get_course_outline", {"course_name": "FastAPI"}, "a"),
                        ("search_course_content", {"query": "vectors"}, "b")),
        text_reply("Done."),
    )

    ask(generator, manager)

    results = fake_llm.calls[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[0]["content"].startswith("Course: Building Web Servers with FastAPI")
    assert "Embeddings turn text into vectors" in results[1]["content"]


def test_unknown_tool_name_is_reported_to_claude(generator, fake_llm, rag):
    fake_llm.script(tool_call("drop_tables", {}), text_reply("Sorry."))

    ask(generator, rag.tool_manager)

    assert fake_llm.calls[1]["messages"][2]["content"][0]["content"] == "Tool 'drop_tables' not found"


def test_api_failure_propagates(generator, fake_llm):
    fake_llm.script(connection_error())

    with pytest.raises(anthropic.APIConnectionError):
        generator.generate_response("q")


# --- Sequential rounds -------------------------------------------------------

def test_outline_then_search_then_answer(generator, fake_llm, seeded_rag):
    manager = seeded_rag.tool_manager
    fake_llm.script(
        tool_call("get_course_outline", {"course_name": "Retrieval"}, "t1"),
        tool_call("search_course_content", {"query": "Embeddings and Vector Search"}, "t2"),
        text_reply("The RAG course covers it."),
    )

    answer = ask(generator, manager, "Which course covers the topic of lesson 1 of the RAG course?")

    assert answer == "The RAG course covers it."
    calls = fake_llm.calls
    assert len(calls) == 3
    assert [c["tool_choice"] for c in calls] == [AUTO, AUTO, NONE]
    assert all(c["tools"] == manager.get_tool_definitions() for c in calls)

    # Round 2 was decided after seeing round 1's real outline
    assert "Embeddings and Vector Search" in calls[1]["messages"][2]["content"][0]["content"]

    final = calls[2]["messages"]
    assert [m["role"] for m in final] == ["user", "assistant", "user", "assistant", "user"]
    (search_result,) = final[4]["content"]
    assert search_result["tool_use_id"] == "t2"
    assert "Embeddings turn text into vectors" in search_result["content"]


def test_round_limit_forces_final_answer(generator, fake_llm):
    tool = RecordingTool()
    fake_llm.script(
        tool_call("lookup", {"step": 1}, "t1"),
        tool_call("lookup", {"step": 2}, "t2"),
        text_reply("Final."),
    )

    answer = ask(generator, make_manager(tool))

    assert answer == "Final."
    assert tool.calls == [{"step": 1}, {"step": 2}]
    assert len(fake_llm.calls) == 3
    assert fake_llm.calls[2]["tool_choice"] == NONE


def test_tool_request_in_final_call_is_not_executed(generator, fake_llm):
    tool = RecordingTool()
    fake_llm.script(
        tool_call("lookup", {"step": 1}, "t1"),
        tool_call("lookup", {"step": 2}, "t2"),
        tool_call("lookup", {"step": 3}, "t3"),
    )

    ask(generator, make_manager(tool))

    assert tool.calls == [{"step": 1}, {"step": 2}]
    assert len(fake_llm.calls) == 3


def test_history_reaches_every_call(generator, fake_llm, seeded_rag):
    manager = seeded_rag.tool_manager
    fake_llm.script(
        tool_call("get_course_outline", {"course_name": "FastAPI"}, "t1"),
        tool_call("search_course_content", {"query": "routes"}, "t2"),
        text_reply("ok"),
    )

    ask(generator, manager, conversation_history="User: earlier question")

    assert len(fake_llm.calls) == 3
    assert all("User: earlier question" in call["system"] for call in fake_llm.calls)


def test_error_string_does_not_stop_the_loop(generator, fake_llm, seeded_rag):
    manager = seeded_rag.tool_manager
    fake_llm.script(
        tool_call("drop_tables", {}, "t1"),
        tool_call("get_course_outline", {"course_name": "FastAPI"}, "t2"),
        text_reply("Recovered."),
    )

    answer = ask(generator, manager)

    assert answer == "Recovered."
    assert [c["tool_choice"] for c in fake_llm.calls] == [AUTO, AUTO, NONE]


# --- Failures and edge cases -------------------------------------------------

def test_tool_exception_is_reported_and_ends_rounds(generator, fake_llm):
    fake_llm.script(tool_call("lookup", {}, "t1"), text_reply("Sorry, the search is unavailable."))

    answer = ask(generator, make_manager(RaisingTool()))

    assert answer == "Sorry, the search is unavailable."
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[1]["tool_choice"] == NONE
    (result,) = fake_llm.calls[1]["messages"][2]["content"]
    assert result["tool_use_id"] == "t1"
    assert result["is_error"] is True
    assert "db down" in result["content"]


def test_exception_in_parallel_call_keeps_other_results(generator, fake_llm):
    ok = RecordingTool("ok")
    fake_llm.script(
        multi_tool_call(("broken", {}, "a"), ("ok", {"x": 1}, "b")),
        text_reply("Partial answer."),
    )

    ask(generator, make_manager(RaisingTool("broken"), ok))

    assert ok.calls == [{"x": 1}]
    results = fake_llm.calls[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[0]["is_error"] is True
    assert "is_error" not in results[1] and results[1]["content"] == "ok result"
    assert fake_llm.calls[1]["tool_choice"] == NONE


def test_truncated_tool_use_is_not_executed(generator, fake_llm):
    tool = RecordingTool()
    cut_off = _message([ToolUseBlock(type="tool_use", id="t1", name="lookup", input={})], "max_tokens")
    fake_llm.script(cut_off)

    ask(generator, make_manager(tool))

    assert tool.calls == []
    assert len(fake_llm.calls) == 1


def test_configurable_round_limit(fake_llm, test_config):
    gen = AIGenerator(api_key="test-key-not-used", model="test-model", max_tool_rounds=1)
    gen.client = fake_llm
    fake_llm.script(tool_call("lookup", {}, "t1"), text_reply("One round."))

    assert ask(gen, make_manager(RecordingTool())) == "One round."
    assert fake_llm.calls[1]["tool_choice"] == NONE

    rag = RAGSystem(dataclasses.replace(test_config, MAX_TOOL_ROUNDS=1))
    assert rag.ai_generator.max_tool_rounds == 1


def test_api_failure_in_second_round_propagates(generator, fake_llm):
    fake_llm.script(tool_call("lookup", {}, "t1"), connection_error())

    with pytest.raises(anthropic.APIConnectionError):
        ask(generator, make_manager(RecordingTool()))
