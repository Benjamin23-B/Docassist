import uuid

from app.rag.reranker import (
    DeterministicReranker,
    RerankerError,
    rerank_chunks,
)
from app.rag.retriever import RetrievedChunk


def _chunk(content: str, *, distance: float = 0.3, **overrides) -> RetrievedChunk:
    defaults = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "filename": "handbook.md",
        "chunk_index": 0,
        "page_number": None,
        "section": None,
        "content": content,
        "distance": distance,
    }
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


def test_deterministic_reranker_promotes_lexical_matches() -> None:
    strong_prior = _chunk(
        "Security badges must be worn at all times in the building.",
        distance=0.05,
    )
    relevant = _chunk(
        "Employees receive twenty-five days of paid leave each year.",
        distance=0.4,
    )

    reranked = DeterministicReranker().rerank(
        [strong_prior, relevant],
        "How many parental leave days do employees get?",
        top_k=2,
    )

    assert reranked[0].chunk_id == relevant.chunk_id
    assert reranked[0].rerank_score is not None
    assert reranked[0].score == reranked[0].rerank_score
    assert reranked[0].rerank_score > (reranked[1].rerank_score or 0.0)


def test_deterministic_reranker_respects_top_k() -> None:
    chunks = [
        _chunk(f"Document {index} mentions leave and days.", distance=0.1 * index)
        for index in range(1, 5)
    ]

    reranked = DeterministicReranker().rerank(chunks, "leave days", top_k=2)

    assert len(reranked) == 2
    assert [chunk.rerank_score is not None for chunk in reranked] == [True, True]


def test_deterministic_reranker_falls_back_to_prior_without_query_terms() -> None:
    first = _chunk("anything", distance=0.1)
    second = _chunk("anything else", distance=0.4)

    reranked = DeterministicReranker().rerank([first, second], "the and of", top_k=2)

    assert [chunk.chunk_id for chunk in reranked] == [first.chunk_id, second.chunk_id]


def test_rerank_chunks_falls_back_when_provider_fails(monkeypatch) -> None:
    def fail() -> None:
        raise RerankerError("provider down")

    monkeypatch.setattr("app.rag.reranker.get_reranker", fail)
    chunks = [_chunk("first", distance=0.1), _chunk("second", distance=0.2)]

    reranked = rerank_chunks(chunks, query="leave", top_k=1)

    assert reranked == chunks[:1]
