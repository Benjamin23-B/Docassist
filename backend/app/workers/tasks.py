from __future__ import annotations

import logging
import random
import uuid
from datetime import UTC, datetime

from app.core.config import get_settings
from app.db.models import Document, DocumentStatus
from app.db.session import SessionLocal
from app.workers.celery_app import celery_app
from app.workers.dlq import push_dead_letter
from app.workers.pipeline import (
    PermanentIngestionError,
    TransientIngestionError,
    ingest_document,
)

logger = logging.getLogger(__name__)
settings = get_settings()

RETRY_BASE_SECONDS = 30
RETRY_MAX_SECONDS = 600


@celery_app.task(
    bind=True,
    name="ingestion.ingest_document",
    max_retries=settings.ingestion_max_retries,
)
def ingest_document_task(self, document_id: str) -> dict[str, str | int]:
    attempt = self.request.retries + 1

    try:
        result = ingest_document(uuid.UUID(document_id))
        return {
            "status": DocumentStatus.READY.value,
            "document_id": str(result.document_id),
            "chunk_count": result.chunk_count,
            "embedding_model": result.embedding_model,
        }

    except TransientIngestionError as exc:
        if self.request.retries >= self.max_retries:
            _fail(
                document_id,
                exc,
                reason="retries_exhausted",
                attempt=attempt,
            )
            return {"status": DocumentStatus.FAILED.value}

        countdown = min(
            RETRY_BASE_SECONDS * (2**self.request.retries),
            RETRY_MAX_SECONDS,
        ) + random.uniform(0, 5)
        logger.warning(
            "Transient ingestion failure for document %s (attempt %d): %s. Retrying in %.0fs",
            document_id,
            attempt,
            exc,
            countdown,
        )
        raise self.retry(exc=exc, countdown=countdown) from exc

    except PermanentIngestionError as exc:
        _fail(document_id, exc, reason="permanent_error", attempt=attempt)
        return {"status": DocumentStatus.FAILED.value}

    except Exception as exc:
        logger.exception("Unexpected ingestion error for document %s", document_id)
        _fail(document_id, exc, reason="unexpected_error", attempt=attempt)
        return {"status": DocumentStatus.FAILED.value}


def _fail(
    document_id: str,
    exc: Exception,
    *,
    reason: str,
    attempt: int,
) -> None:
    message = str(exc)[:2000]

    try:
        with SessionLocal() as db:
            document = db.get(Document, uuid.UUID(document_id))
            if document is not None:
                document.status = DocumentStatus.FAILED.value
                document.error_message = message
                db.commit()
    except Exception:
        logger.exception("Could not mark document %s as failed", document_id)

    push_dead_letter(
        {
            "document_id": document_id,
            "reason": reason,
            "error_type": type(exc).__name__,
            "error": message,
            "attempts": attempt,
            "failed_at": datetime.now(UTC).isoformat(),
        }
    )
