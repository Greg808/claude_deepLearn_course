# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RAG chatbot over course transcripts: FastAPI backend, ChromaDB vector store, Claude with tool use, and a plain HTML/JS frontend served by the same app. Starter codebase for a DeepLearning.AI course.

## Commands

Uses `uv` and Python 3.13+. Always use `uv` (`uv sync`, `uv add`, `uv run`) to install deps and run the server. Run Python files with `uv run python <file>`, not bare `python`. Never call `pip` directly. Needs `ANTHROPIC_API_KEY` in `.env` at the repo root (see `.env.example`).

```bash
uv sync                                                 # install deps
./run.sh                                                # start server on :8000
cd backend && uv run uvicorn app:app --reload --port 8000   # same, manually
```

- UI: http://localhost:8000, API docs: http://localhost:8000/docs
- Tests: `uv run pytest` (all), `uv run pytest -m "not e2e"` (skip browser tests). Offline: Claude is faked, Chroma uses temp dirs, the embedding model must already be cached. E2e needs `uv run playwright install chromium` once. `xfail(strict=True)` tests mark known defects.
- Frontend formatting: Prettier (`npm install` once, then `scripts/frontend-quality.sh` to check, `--fix` to format). No Python linter or formatter is configured. `main.py` at the root is an unused stub.
- Always run the server from `backend/`: `app.py` uses relative paths (`../docs`, `../frontend`, `./chroma_db`).

## Architecture

Backend modules in `backend/` import each other as top-level modules (`from config import config`), so they only resolve with `backend/` as the working directory.

**Ingestion (on startup):** `app.py` startup hook → `RAGSystem.add_course_folder("../docs")` → `DocumentProcessor.process_course_document` → `VectorStore`.
- Course files must follow a fixed format: first lines `Course Title:`, `Course Link:`, `Course Instructor:`, then `Lesson N: <title>` markers, each optionally followed by `Lesson Link:`. Files are read as plain text (the `.pdf`/`.docx` extension filter does not parse those formats).
- Text is chunked by sentences (`CHUNK_SIZE`/`CHUNK_OVERLAP` in `config.py`) and prefixed with lesson context. The prefix differs: the last lesson's chunks get `Course <title> Lesson N content:`, earlier lessons only get `Lesson N content:` on their first chunk.
- The course title is the ID. A course already in ChromaDB is skipped, so edits to a doc are not re-indexed. Delete `backend/chroma_db/` (gitignored) to rebuild.

**Two ChromaDB collections** (`vector_store.py`):
- `course_catalog`: one entry per course (title, instructor, links, lessons as a JSON string in metadata). Used to resolve a fuzzy course name to an exact title via vector search.
- `course_content`: the chunks, filterable by `course_title` and `lesson_number`.
- Embeddings: `all-MiniLM-L6-v2` via sentence-transformers (downloaded on first run).

**Query flow:** `POST /api/query` → `RAGSystem.query` → `AIGenerator.generate_response`.
- Claude gets two tools (`search_tools.py`) and decides itself whether to use one: `search_course_content` (chunks) and `get_course_outline` (title, link and lesson list from `course_catalog`).
- Up to `MAX_TOOL_ROUNDS` (2) sequential tool rounds, each a separate API call that sees earlier results. After the last round, or when a tool raises (sent to Claude as an `is_error` result), one final call keeps the same tools but sets `tool_choice: none`. Tools are never removed mid-conversation, because returned thinking blocks require identical tools and system.
- Sources reach the UI through a side channel: `CourseSearchTool.last_sources` is set during the search, read via `ToolManager.get_last_sources()`, then reset.
- New tools: subclass `Tool`, register in `RAGSystem.__init__`.

**Sessions:** in-memory only (`session_manager.py`), lost on restart. History (last `MAX_HISTORY` exchanges) is passed as text inside the system prompt, not as message turns.

Model, chunk sizes and limits live in `backend/config.py`.
