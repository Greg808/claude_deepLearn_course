from pathlib import Path

import pytest

from conftest import FASTAPI_COURSE, RAG_COURSE, course_text
from document_processor import DocumentProcessor

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


@pytest.fixture
def processor():
    return DocumentProcessor(chunk_size=800, chunk_overlap=100)


def parse(processor, tmp_path, text, name="course.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return processor.process_course_document(str(path))


def test_parses_course_header(processor, tmp_path):
    course, _ = parse(processor, tmp_path, course_text(RAG_COURSE))

    assert course.title == RAG_COURSE["title"]
    assert course.course_link == RAG_COURSE["link"]
    assert course.instructor == RAG_COURSE["instructor"]


def test_missing_link_becomes_none(processor, tmp_path):
    course, _ = parse(processor, tmp_path, course_text({**FASTAPI_COURSE, "link": None}))

    assert course.title == FASTAPI_COURSE["title"]
    assert course.course_link is None
    assert course.instructor == FASTAPI_COURSE["instructor"]


@pytest.mark.xfail(strict=True, reason="Defect: lines 2-3 are always skipped as header, "
                   "so without link and instructor the first lesson marker is lost")
def test_header_without_link_and_instructor_keeps_first_lesson(processor, tmp_path):
    text = "Course Title: Bare Course\n\nLesson 1: First\nFirst content here.\nLesson 2: Second\nSecond content.\n"
    course, _ = parse(processor, tmp_path, text)

    assert course.instructor is None
    assert [l.lesson_number for l in course.lessons] == [1, 2]


def test_lessons_keep_numbers_titles_and_links(processor, tmp_path):
    course, _ = parse(processor, tmp_path, course_text(RAG_COURSE))

    assert [(l.lesson_number, l.title, l.lesson_link) for l in course.lessons] == [
        (n, t, link) for n, t, link, _ in RAG_COURSE["lessons"]
    ]


def test_non_contiguous_lesson_numbers_are_kept(processor, tmp_path):
    course, _ = parse(processor, tmp_path, course_text(FASTAPI_COURSE))

    assert [l.lesson_number for l in course.lessons] == [1, 3]


def test_lesson_without_link_keeps_its_first_content_line(processor, tmp_path):
    _, chunks = parse(processor, tmp_path, course_text(RAG_COURSE))

    lesson_2 = " ".join(c.content for c in chunks if c.lesson_number == 2)
    assert "Long documents are split into chunks." in lesson_2


def test_chunks_are_tagged_and_cover_all_lesson_text(processor, tmp_path):
    _, chunks = parse(processor, tmp_path, course_text(RAG_COURSE))

    assert all(c.course_title == RAG_COURSE["title"] for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for number, _, _, content in RAG_COURSE["lessons"]:
        lesson_text = " ".join(c.content for c in chunks if c.lesson_number == number)
        for sentence in content.split(". "):
            assert sentence.rstrip(".") in lesson_text


def test_link_lines_are_not_indexed_as_content(processor, tmp_path):
    _, chunks = parse(processor, tmp_path, course_text(RAG_COURSE))

    assert not any("Lesson Link:" in c.content for c in chunks)


def test_document_without_lesson_markers_is_chunked_without_lesson(processor, tmp_path):
    text = "Course Title: Loose Notes\nCourse Link: https://example.com/notes\nCourse Instructor: Bob\n\nJust some text. Another sentence here.\n"
    course, chunks = parse(processor, tmp_path, text)

    assert course.lessons == []
    assert chunks and all(c.lesson_number is None for c in chunks)
    assert "Just some text. Another sentence here." in chunks[0].content


def test_invalid_utf8_bytes_are_tolerated(processor, tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes(course_text(RAG_COURSE).replace("Welcome to", "Caf\xe9 welcome to").encode("latin-1"))

    course, chunks = processor.process_course_document(str(path))

    assert course.title == RAG_COURSE["title"]
    assert len(course.lessons) == 3
    assert chunks


class TestChunkText:
    def test_empty_text_gives_no_chunks(self):
        assert DocumentProcessor(100, 20).chunk_text("   ") == []

    def test_chunks_respect_size_limit(self):
        processor = DocumentProcessor(chunk_size=60, chunk_overlap=0)
        text = " ".join(f"Sentence number {i} is here." for i in range(20))

        chunks = processor.chunk_text(text)

        assert len(chunks) > 1
        assert all(len(c) <= 60 for c in chunks)
        assert " ".join(chunks) == text

    def test_overlap_repeats_trailing_sentence(self):
        processor = DocumentProcessor(chunk_size=60, chunk_overlap=30)
        text = " ".join(f"Sentence number {i} is here." for i in range(6))

        chunks = processor.chunk_text(text)

        for current, following in zip(chunks, chunks[1:]):
            last_sentence = current.split(". ")[-1]
            assert following.startswith(last_sentence.rstrip("."))

    @pytest.mark.xfail(strict=True, reason="Defect: after the final chunk, the overlap step "
                       "emits an extra chunk that only repeats its tail")
    def test_no_chunk_merely_repeats_the_previous_tail(self):
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        chunks = processor.chunk_text("Just some text. Another sentence here.")

        assert chunks == ["Just some text. Another sentence here."]

    def test_oversized_sentence_is_kept_whole(self):
        processor = DocumentProcessor(chunk_size=20, chunk_overlap=0)
        long_sentence = "This single sentence is much longer than twenty characters."

        assert processor.chunk_text(long_sentence) == [long_sentence]


@pytest.mark.parametrize("doc", sorted(DOCS_DIR.glob("*.txt")), ids=lambda p: p.name)
def test_real_course_documents_parse_cleanly(processor, doc):
    """Read-only check that the shipped docs still match the parser's format."""
    course, chunks = processor.process_course_document(str(doc))

    numbers = [l.lesson_number for l in course.lessons]
    assert course.title and not course.title.endswith(".txt")
    assert course.course_link.startswith("https://")
    assert numbers and numbers == sorted(set(numbers))
    assert {c.lesson_number for c in chunks} == set(numbers)
