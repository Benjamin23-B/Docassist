import uuid

from sqlalchemy.dialects import postgresql

from app.api.v1.chat import _build_context, _sse
from app.rag.retriever import (
    RetrievedChunk,
    _keyword_search,
    reciprocal_rank_fusion,
)


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


def test_score_prefers_rerank_then_fused_then_similarity() -> None:
    assert _chunk(distance=0.2).score == 0.8
    assert _chunk(distance=0.2, fused_score=0.55).score == 0.55
    assert _chunk(distance=0.2, fused_score=0.55, rerank_score=0.91).score == 0.91
    assert _chunk(distance=0.2, rerank_score=0.91).similarity == 0.8


def test_rrf_promotes_chunks_found_by_both_retrievers() -> None:
    shared_id = uuid.uuid4()
    dense = [
        _chunk(chunk_id=uuid.uuid4(), distance=0.1),
        _chunk(chunk_id=shared_id, distance=0.2),
    ]
    keyword = [
        _chunk(chunk_id=shared_id, distance=1.0, keyword_score=0.9),
        _chunk(chunk_id=uuid.uuid4(), distance=1.0, keyword_score=0.4),
    ]

    fused = reciprocal_rank_fusion(dense, keyword, limit=3, rank_constant=60)

    assert len(fused) == 3
    assert fused[0].chunk_id == shared_id
    assert fused[0].keyword_score == 0.9
    assert fused[0].score == 1.0
    assert all(0.0 <= chunk.score <= 1.0 for chunk in fused)


def test_rrf_keeps_keyword_only_matches_and_respects_limit() -> None:
    keyword_only = _chunk(distance=1.0, keyword_score=0.7)
    dense = [_chunk(distance=0.15), _chunk(distance=0.25)]

    fused = reciprocal_rank_fusion(dense, [keyword_only], limit=2)

    assert len(fused) == 2
    assert keyword_only.chunk_id in {chunk.chunk_id for chunk in fused}
    assert any(chunk.keyword_score == 0.7 for chunk in fused)


def test_rrf_handles_empty_inputs() -> None:
    assert reciprocal_rank_fusion([], [], limit=5) == []


def test_keyword_search_compiles_full_text_query_and_filters() -> None:
    class FakeResult:
        def all(self) -> list:
            return []

    class FakeSession:
        statement = None

        def execute(self, statement):
            self.statement = statement
            return FakeResult()

    db = FakeSession()
    document_id = uuid.uuid4()
    _keyword_search(
        db,  # type: ignore[arg-type]
        user_id=uuid.uuid4(),
        query="leave OR parental",
        doc_scope=[document_id],
        limit=7,
    )

    sql = str(db.statement.compile(dialect=postgresql.dialect()))
    assert "websearch_to_tsquery" in sql
    assert "ts_rank_cd" in sql
    assert "@@" in sql
    assert "chunks.doc_id IN" in sql
    assert "LIMIT" in sql
