import pytest

from conftest import FASTAPI_COURSE, RAG_COURSE
from models import Course, CourseChunk, Lesson
from vector_store import VectorStore

NONE_METADATA = "Defect: Chroma rejects None metadata values, so optional fields cannot be stored"


@pytest.mark.xfail(strict=True, reason=NONE_METADATA)
def test_course_without_link_or_instructor_can_be_stored(store):
    store.add_course_metadata(Course(title="Linkless", lessons=[Lesson(lesson_number=1, title="Only")]))

    assert store.get_existing_course_titles() == ["Linkless"]


@pytest.mark.xfail(strict=True, reason=NONE_METADATA)
def test_chunk_without_lesson_can_be_stored(store):
    store.add_course_content([CourseChunk(content="Loose notes.", course_title="Notes", chunk_index=0)])

    assert store.search("notes").documents == ["Loose notes."]


def test_catalog_lists_ingested_courses(seeded_store):
    assert sorted(seeded_store.get_existing_course_titles()) == sorted([RAG_COURSE["title"], FASTAPI_COURSE["title"]])
    assert seeded_store.get_course_count() == 2


def test_lesson_metadata_round_trips(seeded_store):
    by_title = {m["title"]: m for m in seeded_store.get_all_courses_metadata()}
    rag = by_title[RAG_COURSE["title"]]

    assert rag["course_link"] == RAG_COURSE["link"]
    assert rag["instructor"] == RAG_COURSE["instructor"]
    assert rag["lessons"] == [
        {"lesson_number": n, "lesson_title": t, "lesson_link": link} for n, t, link, _ in RAG_COURSE["lessons"]
    ]


def test_link_lookups(seeded_store):
    title = RAG_COURSE["title"]

    assert seeded_store.get_course_link(title) == RAG_COURSE["link"]
    assert seeded_store.get_lesson_link(title, 1) == "https://example.com/courses/rag/lesson/1"
    assert seeded_store.get_lesson_link(title, 2) is None  # lesson without link
    assert seeded_store.get_lesson_link(title, 99) is None
    assert seeded_store.get_course_link("No Such Course") is None


def test_search_with_partial_course_name_stays_in_that_course(seeded_store):
    results = seeded_store.search("how are requests handled", course_name="FastAPI")

    assert results.error is None and not results.is_empty()
    assert {m["course_title"] for m in results.metadata} == {FASTAPI_COURSE["title"]}


def test_search_with_lesson_filter(seeded_store):
    results = seeded_store.search("vectors", course_name="Retrieval Augmented", lesson_number=1)

    assert not results.is_empty()
    assert {(m["course_title"], m["lesson_number"]) for m in results.metadata} == {(RAG_COURSE["title"], 1)}
    assert "Embeddings turn text into vectors" in results.documents[0]


def test_search_ranks_relevant_content_first(seeded_store):
    results = seeded_store.search("splitting long documents into overlapping pieces")

    assert results.metadata[0]["lesson_number"] == 2


def test_search_honours_result_limit(seeded_store):
    assert len(seeded_store.search("course", limit=2).documents) == 2


def test_search_in_empty_catalog_reports_unknown_course(store):
    results = store.search("anything", course_name="MCP")

    assert results.is_empty()
    assert results.error == "No course found matching 'MCP'"


def test_clear_all_data_empties_both_collections(seeded_store):
    seeded_store.clear_all_data()

    assert seeded_store.get_course_count() == 0
    assert seeded_store.search("vectors").is_empty()


def test_data_survives_a_new_store_instance(seeded_store, test_config):
    reopened = VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL)

    assert reopened.get_course_count() == 2
    assert reopened.get_lesson_link(RAG_COURSE["title"], 0) == "https://example.com/courses/rag/lesson/0"
