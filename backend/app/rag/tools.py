"""Agent tools that read from the user's ingested documents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, DocumentStatus
from app.rag.retriever import RetrievedChunk, retrieve_chunks

SUPPORTED_TOOLS = {
    "search_documents": "Search the user's ready documents for passages relevant to a query.",
    "summarize_document": "Collect representative passages from one or more documents.",
    "compare_documents": "Collect passages about the same question from several documents.",
    "get_page": "Read the indexed chunks for a specific page of a document.",
}


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    matches: int = 0
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status,
            "matches": self.matches,
            "detail": self.detail,
        }


def parse_uuid_ids(values: Any) -> list[uuid.UUID]:
    if not isinstance(values, list):
        raise ValueError("document_ids must be a list of UUIDs")
    parsed: list[uuid.UUID] = []
    for value in values:
        parsed.append(uuid.UUID(str(value)))
    return parsed


def search_documents(
    db: Session,
    *,
    user_id: uuid.UUID,
    query: str,
    document_ids: list[uuid.UUID],
    candidate_k: int,
) -> list[RetrievedChunk]:
    return retrieve_chunks(
        db,
        user_id=user_id,
        query=query,
        doc_scope=document_ids,
        top_k=candidate_k,
    )


def summarize_documents(
    db: Session,
    *,
    user_id: uuid.UUID,
    document_ids: list[uuid.UUID],
    query: str,
    candidate_k: int,
) -> list[RetrievedChunk]:
    if not document_ids:
        raise ValueError("summarize_document requires at least one document id")

    per_document = max(1, candidate_k // len(document_ids))
    collected: list[RetrievedChunk] = []
    for document_id in document_ids:
        chunks = retrieve_chunks(
            db,
            user_id=user_id,
            query=query,
            doc_scope=[document_id],
            top_k=per_document,
        )
        if not chunks:
            # A very specific summary query can miss every chunk; fall back to
            # the opening passages so the answer still has document context.
            chunks = _fetch_document_chunks(
                db,
                user_id=user_id,
                document_id=document_id,
                limit=per_document,
            )
        collected.extend(chunks)
    return collected


def compare_documents(
    db: Session,
    *,
    user_id: uuid.UUID,
    document_ids: list[uuid.UUID],
    query: str,
    candidate_k: int,
) -> list[RetrievedChunk]:
    if len(document_ids) < 2:
        raise ValueError("compare_documents requires at least two document ids")

    per_document = max(1, candidate_k // len(document_ids))
    collected: list[RetrievedChunk] = []
    for document_id in document_ids:
        chunks = retrieve_chunks(
            db,
            user_id=user_id,
            query=query,
            doc_scope=[document_id],
            top_k=per_document,
        )
        if not chunks:
            chunks = _fetch_document_chunks(
                db,
                user_id=user_id,
                document_id=document_id,
                limit=1,
            )
        collected.extend(chunks)
    return collected


def get_page(
    db: Session,
    *,
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    page_number: int,
) -> list[RetrievedChunk]:
    if page_number <= 0:
        raise ValueError("page_number must be positive")
    return _fetch_document_chunks(
        db,
        user_id=user_id,
        document_id=document_id,
        page_number=page_number,
        limit=20,
    )


def _fetch_document_chunks(
    db: Session,
    *,
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    limit: int,
    page_number: int | None = None,
) -> list[RetrievedChunk]:
    statement = (
        select(
            Chunk.id,
            Chunk.doc_id,
            Document.filename,
            Chunk.chunk_index,
            Chunk.page_number,
            Chunk.section,
            Chunk.content,
        )
        .join(Document, Chunk.doc_id == Document.id)
        .where(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY.value,
            Chunk.doc_id == document_id,
        )
        .order_by(Chunk.chunk_index)
        .limit(limit)
    )
    if page_number is not None:
        statement = statement.where(Chunk.page_number == page_number)

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
            # These are context-gathering tools, not similarity searches.
            distance=1.0,
        )
        for row in rows
    ]
