"""Retrieval over a user's ingested chunks.

Dense vector search and PostgreSQL full-text search each produce a ranked list;
reciprocal rank fusion merges them without needing comparable score scales.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Chunk, Document, DocumentStatus
from app.workers.embedder import get_embedding_provider

SNIPPET_LENGTH = 280


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    page_number: int | None
    section: str | None
    content: str
    distance: float
    keyword_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    @property
    def similarity(self) -> float:
        """Cosine similarity in [-1, 1]; higher is closer."""
        return round(1.0 - self.distance, 4)

    @property
    def score(self) -> float:
        """Best available relevance score in [0, 1]."""
        if self.rerank_score is not None:
            return round(self.rerank_score, 4)
        if self.fused_score is not None:
            return round(self.fused_score, 4)
        return self.similarity

    @property
    def snippet(self) -> str:
        text = " ".join(self.content.split())
        if len(text) <= SNIPPET_LENGTH:
            return text
        return f"{text[:SNIPPET_LENGTH].rsplit(' ', 1)[0]}…"


def retrieve_chunks(
    db: Session,
    *,
    user_id: uuid.UUID,
    query: str,
    doc_scope: list[uuid.UUID] | tuple[uuid.UUID, ...] = (),
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Return the most relevant chunks for ``query`` across the user's ready docs."""
    settings = get_settings()
    limit = top_k or settings.retrieval_top_k
    candidate_limit = limit
    if settings.hybrid_search_enabled:
        candidate_limit = max(limit, limit * settings.hybrid_candidate_multiplier)

    dense = _vector_search(
        db,
        user_id=user_id,
        query=query,
        doc_scope=doc_scope,
        limit=candidate_limit,
    )
    if not settings.hybrid_search_enabled:
        return dense[:limit]

    keyword = _keyword_search(
        db,
        user_id=user_id,
        query=query,
        doc_scope=doc_scope,
        limit=candidate_limit,
    )
    if not keyword:
        return dense[:limit]

    return reciprocal_rank_fusion(
        dense,
        keyword,
        limit=limit,
        rank_constant=settings.hybrid_rrf_k,
        vector_weight=settings.hybrid_vector_weight,
        keyword_weight=settings.hybrid_keyword_weight,
    )


def reciprocal_rank_fusion(
    dense: Sequence[RetrievedChunk],
    keyword: Sequence[RetrievedChunk],
    *,
    limit: int,
    rank_constant: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Merge two ranked lists with weighted reciprocal rank fusion."""
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    base: dict[uuid.UUID, RetrievedChunk] = {}
    raw_scores: dict[uuid.UUID, float] = {}
    keyword_scores: dict[uuid.UUID, float | None] = {}

    for rank, chunk in enumerate(dense, start=1):
        base.setdefault(chunk.chunk_id, chunk)
        raw_scores[chunk.chunk_id] = raw_scores.get(chunk.chunk_id, 0.0) + (
            vector_weight / (rank_constant + rank)
        )

    for rank, chunk in enumerate(keyword, start=1):
        base.setdefault(chunk.chunk_id, chunk)
        raw_scores[chunk.chunk_id] = raw_scores.get(chunk.chunk_id, 0.0) + (
            keyword_weight / (rank_constant + rank)
        )
        keyword_scores[chunk.chunk_id] = chunk.keyword_score

    if not raw_scores:
        return []

    max_raw = max(raw_scores.values())
    fused = [
        replace(
            chunk,
            keyword_score=keyword_scores.get(chunk.chunk_id, chunk.keyword_score),
            fused_score=(raw_scores[chunk.chunk_id] / max_raw) if max_raw else 0.0,
        )
        for chunk_id, chunk in base.items()
    ]
    fused.sort(
        key=lambda chunk: (chunk.fused_score or 0.0, chunk.similarity),
        reverse=True,
    )
    return fused[:limit]


def _vector_search(
    db: Session,
    *,
    user_id: uuid.UUID,
    query: str,
    doc_scope: Sequence[uuid.UUID],
    limit: int,
) -> list[RetrievedChunk]:
    query_vector = get_embedding_provider().embed([query])[0]
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")

    statement = (
        select(
            Chunk.id,
            Chunk.doc_id,
            Document.filename,
            Chunk.chunk_index,
            Chunk.page_number,
            Chunk.section,
            Chunk.content,
            distance,
        )
        .join(Document, Chunk.doc_id == Document.id)
        .where(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY.value,
            Chunk.embedding.is_not(None),
        )
        .order_by(distance)
        .limit(limit)
    )
    if doc_scope:
        statement = statement.where(Chunk.doc_id.in_(list(doc_scope)))

    rows = db.execute(statement).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.doc_id,
            filename=row.filename,
            chunk_index=row.chunk_index,
            page_number=row.page_number,
            section=row.section,
            content=row.content,
            distance=float(row.distance),
        )
        for row in rows
    ]


def _keyword_search(
    db: Session,
    *,
    user_id: uuid.UUID,
    query: str,
    doc_scope: Sequence[uuid.UUID],
    limit: int,
) -> list[RetrievedChunk]:
    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(Chunk.tsv, tsquery).label("keyword_score")

    statement = (
        select(
            Chunk.id,
            Chunk.doc_id,
            Document.filename,
            Chunk.chunk_index,
            Chunk.page_number,
            Chunk.section,
            Chunk.content,
            rank,
        )
        .join(Document, Chunk.doc_id == Document.id)
        .where(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY.value,
            Chunk.tsv.op("@@")(tsquery),
        )
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    if doc_scope:
        statement = statement.where(Chunk.doc_id.in_(list(doc_scope)))

    rows = db.execute(statement).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.doc_id,
            filename=row.filename,
            chunk_index=row.chunk_index,
            page_number=row.page_number,
            section=row.section,
            content=row.content,
            # No cosine distance for a keyword-only match; the fused score is
            # what ranks it, and `similarity` deliberately reads as zero.
            distance=1.0,
            keyword_score=float(row.keyword_score),
        )
        for row in rows
    ]
