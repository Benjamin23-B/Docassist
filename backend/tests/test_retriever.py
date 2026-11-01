import uuid

from app.api.v1.chat import _build_context, _sse
from app.rag.retriever import RetrievedChunk


def _chunk(**overrides) -> RetrievedChunk:
    defaults = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "filename": "handbook.md",
        "chunk_index": 3,
        "page_number": 2,
        "section": "Leave Policy",
        "content": (
            "Employees receive twenty-five days of paid leave each year. "
            "Requests are approved by managers."
        ),
        "distance": 0.25,
    }
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


def test_score_is_cosine_similarity() -> None:
    chunk = _chunk(distance=0.25)
    assert chunk.score == 0.75


def test_snippet_is_whitespace_normalised_and_truncated() -> None:
    content = "word " * 100
    snippet = _chunk(content=content).snippet

    assert len(snippet) <= 281
    assert "\n" not in snippet
    assert snippet.endswith("…")


def test_build_context_labels_sources_and_citations() -> None:
    chunk = _chunk()
    passages, citations = _build_context([chunk])

    assert len(passages) == 1
    assert passages[0].index == 1
    assert "handbook.md" in passages[0].source
    assert "page 2" in passages[0].source
    assert "Leave Policy" in passages[0].source

    citation = citations[0]
    assert citation.index == 1
    assert citation.document_id == chunk.document_id
    assert citation.chunk_id == chunk.chunk_id
    assert citation.page_number == 2
    assert citation.section == "Leave Policy"
    assert citation.score == 0.75


def test_build_context_handles_documents_without_pages() -> None:
    passages, citations = _build_context([_chunk(page_number=None, section=None)])

    assert passages[0].source == "handbook.md"
    assert citations[0].page_number is None
    assert citations[0].section is None


def test_sse_frames_are_valid_event_stream() -> None:
    frame = _sse("token", {"text": "hello"})

    assert frame.startswith("event: token\n")
    assert '"text": "hello"' in frame
    assert frame.endswith("\n\n")
