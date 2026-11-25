import anthropic
import pytest

from conftest import FASTAPI_COURSE, RAG_COURSE, connection_error, text_reply, tool_call


class TestIngestion:
    def test_folder_ingestion_counts_courses_and_chunks(self, rag, course_dir):
        courses, chunks = rag.add_course_folder(str(course_dir))

        assert courses == 2
        assert chunks == rag.vector_store.course_content.count() > 0

    def test_only_supported_extensions_are_read(self, rag, course_dir):
        (course_dir / "notes.md").write_text("Course Title: Markdown Notes\n", encoding="utf-8")

        rag.add_course_folder(str(course_dir))

        assert "Markdown Notes" not in rag.vector_store.get_existing_course_titles()

    def test_rerun_skips_known_courses(self, seeded_rag, course_dir):
        chunks_before = seeded_rag.vector_store.course_content.count()

        assert seeded_rag.add_course_folder(str(course_dir)) == (0, 0)
        assert seeded_rag.vector_store.course_content.count() == chunks_before

    def test_clear_existing_rebuilds(self, seeded_rag, course_dir):
        courses, _ = seeded_rag.add_course_folder(str(course_dir), clear_existing=True)

        assert courses == 2
        assert seeded_rag.vector_store.get_course_count() == 2

    def test_missing_folder(self, rag, tmp_path):
        assert rag.add_course_folder(str(tmp_path / "nope")) == (0, 0)

    def test_analytics(self, seeded_rag):
        analytics = seeded_rag.get_course_analytics()

        assert analytics["total_courses"] == 2
        assert sorted(analytics["course_titles"]) == sorted([RAG_COURSE["title"], FASTAPI_COURSE["title"]])


class TestQuery:
    def test_outline_question_end_to_end(self, seeded_rag, fake_llm):
        fake_llm.script(tool_call("get_course_outline", {"course_name": "Retrieval Augmented Generation course"}), text_reply("Three lessons."))

        answer, sources = seeded_rag.query("What is the outline of the RAG course?")

        tool_output = fake_llm.calls[1]["messages"][2]["content"][0]["content"]
        assert tool_output.splitlines()[:2] == [f"Course: {RAG_COURSE['title']}", f"Link: {RAG_COURSE['link']}"]
        assert [l for l in tool_output.splitlines() if l.startswith("- Lesson")] == [
            f"- Lesson {n}: {t}" for n, t, _, _ in RAG_COURSE["lessons"]
        ]
        assert answer == "Three lessons."
        assert sources == [{"text": RAG_COURSE["title"], "url": RAG_COURSE["link"]}]

    def test_content_question_cites_lesson_links(self, seeded_rag, fake_llm):
        fake_llm.script(
            tool_call("search_course_content", {"query": "dependencies", "course_name": "FastAPI", "lesson_number": 3}),
            text_reply("Declared as parameters."),
        )

        _, sources = seeded_rag.query("How does dependency injection work?")

        assert {(s["text"], s["url"]) for s in sources} == {
            (f"{FASTAPI_COURSE['title']} - Lesson 3", "https://example.com/courses/fastapi/lesson/3")
        }

    def test_sources_do_not_leak_into_next_answer(self, seeded_rag, fake_llm):
        fake_llm.script(tool_call("get_course_outline", {"course_name": "FastAPI"}), text_reply("a"), text_reply("b"))

        seeded_rag.query("outline?")
        _, sources = seeded_rag.query("What is 2+2?")

        assert sources == []

    @pytest.mark.xfail(strict=True, reason="Defect: sources are only reset after a successful "
                       "answer, so a failed query leaks its sources into the next one")
    def test_failed_query_does_not_leak_sources(self, seeded_rag, fake_llm):
        sm = seeded_rag.session_manager
        fake_llm.script(tool_call("get_course_outline", {"course_name": "FastAPI"}), connection_error(), text_reply("4"))

        with pytest.raises(anthropic.APIConnectionError):
            seeded_rag.query("outline?", sm.create_session())
        _, sources = seeded_rag.query("What is 2+2?", sm.create_session())

        assert sources == []


class TestSessions:
    def test_history_stays_within_its_session(self, rag, fake_llm):
        a, b = rag.session_manager.create_session(), rag.session_manager.create_session()
        fake_llm.script(text_reply("Answer for A"), text_reply("A again"), text_reply("Answer for B"))

        rag.query("question from A", a)
        rag.query("follow-up from A", a)
        rag.query("question from B", b)

        assert "question from A" in fake_llm.calls[1]["system"]
        assert "Answer for A" in fake_llm.calls[1]["system"]
        assert "from A" not in fake_llm.calls[2]["system"]
        assert "Answer for" not in fake_llm.calls[2]["system"]

    def test_history_keeps_only_recent_exchanges(self, rag, fake_llm):
        session = rag.session_manager.create_session()
        fake_llm.script(*(text_reply(f"answer {i}") for i in range(4)))

        for i in range(4):
            rag.query(f"question {i}", session)

        last_system = fake_llm.calls[3]["system"]
        assert rag.config.MAX_HISTORY == 2
        assert "question 0" not in last_system
        assert "question 1" in last_system and "question 2" in last_system

    def test_query_without_session_records_nothing(self, rag, fake_llm):
        fake_llm.script(text_reply("x"))

        rag.query("hi")

        assert rag.session_manager.sessions == {}

    def test_failed_query_records_no_exchange(self, rag, fake_llm):
        session = rag.session_manager.create_session()
        fake_llm.script(connection_error())

        with pytest.raises(anthropic.APIConnectionError):
            rag.query("lost question", session)

        assert rag.session_manager.get_conversation_history(session) is None

    def test_session_ids_are_unique(self, rag):
        ids = {rag.session_manager.create_session() for _ in range(5)}

        assert len(ids) == 5
