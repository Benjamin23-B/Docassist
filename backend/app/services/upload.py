"""Upload validation and streaming helpers."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.core.config import Settings

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Declared content types accepted for each supported extension. Browsers often
# send `application/octet-stream` for markdown/plain text, which is handled as
# a generic type below.
EXTENSION_MIME_TYPES: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain"},
    ".markdown": {"text/markdown", "text/x-markdown", "text/plain"},
    ".txt": {"text/plain"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".html": {"text/html", "application/xhtml+xml"},
    ".htm": {"text/html", "application/xhtml+xml"},
}

GENERIC_MIME_TYPES = {"", "application/octet-stream", "binary/octet-stream"}

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._ -]")


def sanitize_filename(filename: str | None) -> str:
    """Reduce a client-supplied filename to a safe basename.

    Strips directory components (both POSIX and Windows separators) and any
    character that could break object keys or response headers.
    """
    if not filename or not filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required",
        )

    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    basename = _UNSAFE_FILENAME_CHARS.sub("_", basename)

    if not basename or not basename.strip("."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename",
        )

    stem, suffix = os.path.splitext(basename)
    stem = stem[:180].strip() or "document"
    return f"{stem}{suffix[:20]}"


def validate_upload(
    filename: str | None,
    content_type: str | None,
    settings: Settings,
) -> tuple[str, str]:
    """Validate a prospective upload and return `(filename, mime_type)`."""
    clean_name = sanitize_filename(filename)
    suffix = Path(clean_name).suffix.lower()

    if suffix not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type {suffix or '(none)'}. Allowed: {allowed}",
        )

    declared = (content_type or "").split(";")[0].strip().lower()
    expected = EXTENSION_MIME_TYPES.get(suffix, set())

    if declared not in GENERIC_MIME_TYPES and expected and declared not in expected:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(f"Content type {declared!r} does not match the {suffix} extension"),
        )

    if declared in GENERIC_MIME_TYPES:
        resolved = sorted(expected)[0] if expected else "application/octet-stream"
    else:
        resolved = declared

    return clean_name, resolved


def spool_upload(file: UploadFile, settings: Settings) -> tuple[Path, str, int]:
    """Stream an upload to a temp file while hashing and enforcing the size cap.

    Returns `(temp_path, sha256_hexdigest, size_in_bytes)`. The caller owns the
    temp file and must remove it.
    """
    hasher = hashlib.sha256()
    size = 0
    tmp = tempfile.NamedTemporaryFile(prefix="rag-upload-", suffix=".part", delete=False)

    try:
        while True:
            chunk = file.file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > settings.max_upload_size_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File exceeds the maximum upload size of {settings.max_upload_size_mb} MB"
                    ),
                )
            hasher.update(chunk)
            tmp.write(chunk)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
    except BaseException:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise

    if size == 0:
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty",
        )

    return Path(tmp.name), hasher.hexdigest(), size


def build_object_key(
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    filename: str,
) -> str:
    return f"users/{user_id}/documents/{document_id}/{filename}"
