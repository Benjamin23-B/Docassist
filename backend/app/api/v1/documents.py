from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, DbSession
from app.core.config import get_settings
from app.db.models import Chunk, Document, DocumentStatus
from app.schemas.document import DocumentDetail, DocumentListResponse, DocumentRead
from app.services.storage import StorageError, StorageService, get_storage
from app.services.upload import (
    build_object_key,
    spool_upload,
    validate_upload,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

StorageDep = Annotated[StorageService, Depends(get_storage)]


def _enqueue_ingestion(document_id: uuid.UUID) -> None:
    """Hand the document to the Celery ingestion worker.

    A broker outage must not fail the upload itself: the document stays
    ``queued`` and can be retried through the reindex endpoint.
    """
    try:
        from app.workers.tasks import ingest_document_task

        ingest_document_task.delay(str(document_id))
    except Exception:
        logger.exception(
            "Could not enqueue ingestion for document %s; use "
            "POST /api/v1/documents/%s/reindex to retry",
            document_id,
            document_id,
        )


def _get_owned_document(
    document_id: uuid.UUID,
    db: Session,
    user_id: uuid.UUID,
) -> Document:
    document = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
    )
    if document is None:
        # Deliberately 404 for other users' documents as well, so the API does
        # not leak the existence of resources across tenants.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return document


@router.post(
    "",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
    responses={
        409: {"description": "Identical file already uploaded"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
        502: {"description": "Object storage unavailable"},
    },
)
def upload_document(
    file: Annotated[
        UploadFile,
        File(description="Document to upload (PDF, Markdown, TXT, DOCX, HTML)"),
    ],
    db: DbSession,
    current_user: CurrentUser,
    storage: StorageDep,
) -> Document:
    filename, mime_type = validate_upload(file.filename, file.content_type, settings)

    tmp_path: Path | None = None
    uploaded_key: str | None = None
    document: Document | None = None
    document_id = uuid.uuid4()

    try:
        tmp_path, file_hash, size_bytes = spool_upload(file, settings)

        duplicate = db.scalar(
            select(Document).where(
                Document.user_id == current_user.id,
                Document.file_hash == file_hash,
            )
        )
        if duplicate is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This file has already been uploaded",
                headers={"X-Document-Id": str(duplicate.id)},
            )

        document = Document(
            id=document_id,
            user_id=current_user.id,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            file_hash=file_hash,
            storage_url="",
            status=DocumentStatus.QUEUED.value,
        )
        uploaded_key = build_object_key(current_user.id, document_id, filename)
        document.storage_url = f"s3://{storage.bucket}/{uploaded_key}"

        with tmp_path.open("rb") as fileobj:
            storage.upload_fileobj(fileobj, uploaded_key, mime_type)

        db.add(document)
        db.commit()
        db.refresh(document)

    except HTTPException:
        db.rollback()
        raise
    except StorageError as exc:
        db.rollback()
        logger.exception("Object storage upload failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not store the uploaded file. Please try again.",
        ) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        if uploaded_key is not None:
            try:
                storage.delete_object(uploaded_key)
            except StorageError:
                logger.warning("Could not clean up orphaned object %s", uploaded_key)
        logger.exception("Failed to persist document metadata")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save document metadata",
        ) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        file.file.close()

    if document is None:  # pragma: no cover - defensive, set on every success path
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document creation failed",
        )
    _enqueue_ingestion(document.id)
    return document


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents for the current user",
)
def list_documents(
    db: DbSession,
    current_user: CurrentUser,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    status_filter: Annotated[
        DocumentStatus | None,
        Query(alias="status", description="Filter by processing status"),
    ] = None,
) -> DocumentListResponse:
    filters = [Document.user_id == current_user.id]
    if status_filter is not None:
        filters.append(Document.status == status_filter.value)

    total = db.scalar(select(func.count()).select_from(Document).where(*filters)) or 0
    documents = db.scalars(
        select(Document)
        .where(*filters)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()

    return DocumentListResponse(
        items=list(documents),
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    summary="Get document details and processing status",
)
def get_document(
    document_id: uuid.UUID,
    db: DbSession,
    current_user: CurrentUser,
) -> DocumentDetail:
    document = _get_owned_document(document_id, db, current_user.id)
    chunk_count = (
        db.scalar(select(func.count()).select_from(Chunk).where(Chunk.doc_id == document.id)) or 0
    )
    detail = DocumentDetail.model_validate(document)
    detail.chunk_count = chunk_count
    return detail


@router.post(
    "/{document_id}/reindex",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-run ingestion (re-parse, re-chunk and re-embed)",
    responses={409: {"description": "Stored file is missing"}},
)
def reindex_document(
    document_id: uuid.UUID,
    db: DbSession,
    current_user: CurrentUser,
    storage: StorageDep,
) -> Document:
    document = _get_owned_document(document_id, db, current_user.id)
    if storage.key_from_url(document.storage_url) is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is not stored in this object storage bucket",
        )

    document.status = DocumentStatus.QUEUED.value
    document.error_message = None
    db.commit()
    db.refresh(document)
    _enqueue_ingestion(document.id)
    return document


@router.get(
    "/{document_id}/download",
    summary="Redirect to a short-lived download URL",
    responses={307: {"description": "Redirect to object storage"}},
)
def download_document(
    document_id: uuid.UUID,
    db: DbSession,
    current_user: CurrentUser,
    storage: StorageDep,
) -> RedirectResponse:
    document = _get_owned_document(document_id, db, current_user.id)
    key = storage.key_from_url(document.storage_url)
    if key is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is not stored in this object storage bucket",
        )

    try:
        url = storage.presigned_download_url(key, document.filename)
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate a download link",
        ) from exc

    return RedirectResponse(url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its stored file",
)
def delete_document(
    document_id: uuid.UUID,
    db: DbSession,
    current_user: CurrentUser,
    storage: StorageDep,
) -> Response:
    document = _get_owned_document(document_id, db, current_user.id)
    key = storage.key_from_url(document.storage_url)

    db.delete(document)
    db.commit()

    if key is not None:
        try:
            storage.delete_object(key)
        except StorageError:
            # Metadata is already gone; a leftover object can be reclaimed by a
            # later cleanup job rather than failing the request.
            logger.warning(
                "Could not delete object %s for removed document %s",
                key,
                document_id,
            )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
