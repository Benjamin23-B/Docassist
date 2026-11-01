"""Document ingestion pipeline: download -> parse -> chunk -> embed -> index."""

from __future__ import annotations

import logging
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, insert
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Chunk, Document, DocumentStatus
from app.db.session import SessionLocal
from app.services.storage import StorageError, get_storage
from app.workers.chunker import Chunk as TextChunk
from app.workers.chunker import chunk_blocks, count_tokens
from app.workers.embedder import (
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingRequestError,
    get_embedding_provider,
)
from app.workers.parser import DocumentParseError, ParsedBlock, parse_file

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Base class for ingestion failures."""


class PermanentIngestionError(IngestionError):
    """Failure that will not succeed on retry (bad input, missing config)."""


class TransientIngestionError(IngestionError):
    """Failure worth retrying (object storage or provider unavailable)."""


@dataclass(slots=True)
class IngestionResult:
    document_id: uuid.UUID
    chunk_count: int
    token_count: int
    embedding_model: str


def ingest_document(document_id: uuid.UUID) -> IngestionResult:
    """Run the full pipeline for one document. Safe to call repeatedly."""
    settings = get_settings()

    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if document is None:
            raise PermanentIngestionError(f"Document {document_id} does not exist")

        document.status = DocumentStatus.PROCESSING.value
        document.error_message = None
        db.commit()

        try:
            blocks = _parse_document(document)
            chunks = chunk_blocks(
                blocks,
                max_tokens=settings.chunk_size_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                min_tokens=settings.min_chunk_tokens,
            )
            if not chunks:
                raise PermanentIngestionError(
                    "No extractable text found in document (scanned PDFs require OCR)"
                )

            provider = get_embedding_provider()
            vectors = _embed(provider, chunks)
            _replace_chunks(db, document.id, chunks, vectors, provider.model_name)

            document.status = DocumentStatus.READY.value
            document.error_message = None
            db.commit()

            result = IngestionResult(
                document_id=document.id,
                chunk_count=len(chunks),
                token_count=sum(count_tokens(chunk.content) for chunk in chunks),
                embedding_model=provider.model_name,
            )
            logger.info(
                "Ingested document %s: %d chunks, %d tokens (%s)",
                document.id,
                result.chunk_count,
                result.token_count,
                provider.model_name,
            )
            return result

        except PermanentIngestionError as exc:
            _mark_failed(db, document.id, exc)
            raise
        except (DocumentParseError, EmbeddingProviderError) as exc:
            error = PermanentIngestionError(str(exc))
            _mark_failed(db, document.id, error)
            raise error from exc
        except EmbeddingRequestError as exc:
            db.rollback()
            raise TransientIngestionError(str(exc)) from exc
        except (StorageError, OperationalError, InterfaceError) as exc:
            db.rollback()
            raise TransientIngestionError(str(exc)) from exc
        except Exception as exc:
            error = PermanentIngestionError(f"Unexpected ingestion failure: {exc}")
            _mark_failed(db, document.id, error)
            raise error from exc


def _parse_document(document: Document) -> list[ParsedBlock]:
    storage = get_storage()
    key = storage.key_from_url(document.storage_url)
    if key is None:
        raise PermanentIngestionError(
            "Document is not stored in the configured object storage bucket"
        )

    suffix = Path(document.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(
        prefix="rag-ingest-", suffix=suffix, delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        storage.download_to_file(key, temp_path)
        parsed = parse_file(temp_path, suffix)
    finally:
        temp_path.unlink(missing_ok=True)

    return parsed.blocks


def _embed(provider: EmbeddingProvider, chunks: list[TextChunk]) -> list[list[float]]:
    vectors = provider.embed([chunk.content for chunk in chunks])
    if len(vectors) != len(chunks):
        raise EmbeddingProviderError(
            f"Embedding provider returned {len(vectors)} vectors for {len(chunks)} chunks"
        )
    return vectors


def _replace_chunks(
    db: Session,
    document_id: uuid.UUID,
    chunks: list[TextChunk],
    vectors: list[list[float]],
    model_name: str,
) -> None:
    db.execute(delete(Chunk).where(Chunk.doc_id == document_id))
    rows = [
        {
            "doc_id": document_id,
            "chunk_index": chunk.index,
            "page_number": chunk.page_number,
            "section": chunk.section,
            "content": chunk.content,
            "embedding": vector,
            "model_name": model_name,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    if rows:
        db.execute(insert(Chunk), rows)


def _mark_failed(db: Session, document_id: uuid.UUID, exc: Exception) -> None:
    message = str(exc)[:2000]
    try:
        db.rollback()
        document = db.get(Document, document_id)
        if document is None:
            return
        document.status = DocumentStatus.FAILED.value
        document.error_message = message
        db.commit()
    except Exception:
        logger.exception("Could not record failure for document %s", document_id)
