import pytest

from conftest import FASTAPI_COURSE, RAG_COURSE
from models import Course, Lesson
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager


class TestCourseSearchTool:
    def test_results_carry_course_and_lesson_headers(self, seeded_store):
        output = CourseSearchTool(seeded_store).execute("vectors", course_name="Retrieval", lesson_number=1)

        assert output.startswith(f"[{RAG_COURSE['title']} - Lesson 1]\n")
        assert "Embeddings turn text into vectors" in output

    def test_sources_use_lesson_link(self, seeded_store):
        tool = CourseSearchTool(seeded_store)
        tool.execute("vectors", course_name="Retrieval", lesson_number=1)

        assert tool.last_sources and set(map(tuple, (s.values() for s in tool.last_sources))) == {
            (f"{RAG_COURSE['title']} - Lesson 1", "https://example.com/courses/rag/lesson/1")
        }

    def test_sources_fall_back_to_course_link_for_lesson_without_link(self, seeded_store):
        tool = CourseSearchTool(seeded_store)
        tool.execute("chunks", course_name="Retrieval", lesson_number=2)

        assert {s["url"] for s in tool.last_sources} == {RAG_COURSE["link"]}

    def test_empty_result_names_the_filters(self, seeded_store):
        output = CourseSearchTool(seeded_store).execute("anything", course_name="FastAPI", lesson_number=2)

        assert output == "No relevant content found in course 'FastAPI' in lesson 2."

    def test_unknown_course_returns_no_results_from_other_courses(self, seeded_store):
        tool = CourseSearchTool(seeded_store)

        assert tool.execute("vectors", course_name="Medieval Pottery Glazing") == \
            "No course found matching 'Medieval Pottery Glazing'"
        assert tool.last_sources == []

    def test_store_error_is_returned_as_text(self, store):
        assert CourseSearchTool(store).execute("x", course_name="MCP") == "No course found matching 'MCP'"


class TestCourseOutlineTool:
    def test_partial_name_returns_full_outline(self, seeded_store):
        output = CourseOutlineTool(seeded_store).execute(course_name="Retrieval Augmented")
        lines = output.splitlines()

        assert lines[0] == f"Course: {RAG_COURSE['title']}"
        assert lines[1] == f"Link: {RAG_COURSE['link']}"
        assert lines[3:] == [f"- Lesson {n}: {t}" for n, t, _, _ in RAG_COURSE["lessons"]]

    def test_keeps_original_lesson_numbers(self, seeded_store):
        output = CourseOutlineTool(seeded_store).execute(course_name="FastAPI")

        assert "- Lesson 1: Routing Basics" in output
        assert "- Lesson 3: Dependency Injection" in output
        assert "Lesson 2" not in output

    @pytest.mark.xfail(strict=True, reason="Defect: a course without link cannot be stored (None metadata)")
    def test_course_without_link(self, store):
        store.add_course_metadata(Course(title="Linkless Course", lessons=[Lesson(lesson_number=1, title="Only")]))

        assert CourseOutlineTool(store).execute(course_name="Linkless").splitlines()[1] == "Link: n/a"

    def test_sets_course_as_source(self, seeded_store):
        tool = CourseOutlineTool(seeded_store)
        tool.execute(course_name="Retrieval")

        assert tool.last_sources == [{"text": RAG_COURSE["title"], "url": RAG_COURSE["link"]}]

    def test_empty_catalog(self, store):
        tool = CourseOutlineTool(store)

        assert tool.execute(course_name="MCP") == "No course found matching 'MCP'"
        assert tool.last_sources == []

    @pytest.mark.parametrize("name", ["Medieval Pottery Glazing", "Python", "Introduction to Cooking"])
    def test_unknown_course_is_not_matched_to_another(self, seeded_store, name):
        tool = CourseOutlineTool(seeded_store)

        assert tool.execute(course_name=name) == f"No course found matching '{name}'"
        assert tool.last_sources == []

    @pytest.mark.parametrize("name", ["MCP", "mcp course"])
    def test_abbreviation_in_title_still_matches(self, store, name):
        title = "MCP: Build Rich-Context AI Apps with Anthropic"
        store.add_course_metadata(Course(title=title, course_link="https://example.com/mcp", instructor="x",
                                         lessons=[Lesson(lesson_number=0, title="Introduction")]))
        store.add_course_metadata(Course(title=RAG_COURSE["title"], course_link=RAG_COURSE["link"], instructor="x"))

        assert CourseOutlineTool(store).execute(course_name=name).splitlines()[0] == f"Course: {title}"


class TestToolManager:
    def test_rag_system_registers_both_tools(self, rag):
        names = [d["name"] for d in rag.tool_manager.get_tool_definitions()]

        assert sorted(names) == ["get_course_outline", "search_course_content"]

    def test_definitions_are_valid_tool_schemas(self, rag):
        for definition in rag.tool_manager.get_tool_definitions():
            schema = definition["input_schema"]
            assert definition["description"]
            assert schema["type"] == "object"
            assert set(schema["required"]) <= set(schema["properties"])

    def test_dispatches_by_name(self, seeded_rag):
        output = seeded_rag.tool_manager.execute_tool("get_course_outline", course_name="FastAPI")

        assert output.startswith(f"Course: {FASTAPI_COURSE['title']}")

    def test_unknown_tool(self):
        assert ToolManager().execute_tool("delete_everything") == "Tool 'delete_everything' not found"

    def test_tool_without_name_is_rejected(self):
        class Nameless(CourseSearchTool):
            def get_tool_definition(self):
                return {"description": "x"}

        with pytest.raises(ValueError):
            ToolManager().register_tool(Nameless(None))

    def test_last_sources_come_from_the_tool_that_ran_and_reset(self, seeded_rag):
        manager = seeded_rag.tool_manager
        manager.execute_tool("get_course_outline", course_name="Retrieval")

        assert manager.get_last_sources() == [{"text": RAG_COURSE["title"], "url": RAG_COURSE["link"]}]
        manager.reset_sources()
        assert manager.get_last_sources() == []
