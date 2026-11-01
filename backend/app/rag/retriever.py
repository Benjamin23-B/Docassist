"""Vector retrieval over a user's ingested chunks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
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

    @property
    def score(self) -> float:
        """Cosine similarity in [-1, 1]; higher is closer."""
        return round(1.0 - self.distance, 4)

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
    """Return the nearest chunks for ``query`` across the user's ready docs."""
    settings = get_settings()
    limit = top_k or settings.retrieval_top_k

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
