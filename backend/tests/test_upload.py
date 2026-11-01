import pytest
from fastapi import HTTPException

from app.core.config import get_settings
from app.services.upload import sanitize_filename, validate_upload

settings = get_settings()


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\notes.txt", "notes.txt"),
        ("weird name<script>.md", "weird name_script_.md"),
        ("  spaced.txt  ", "spaced.txt"),
    ],
)
def test_sanitize_filename(filename: str, expected: str) -> None:
    assert sanitize_filename(filename) == expected


def test_sanitize_filename_rejects_empty_name() -> None:
    with pytest.raises(HTTPException) as excinfo:
        sanitize_filename("   ")
    assert excinfo.value.status_code == 400


def test_sanitize_filename_rejects_dot_only_name() -> None:
    with pytest.raises(HTTPException):
        sanitize_filename("..")


def test_validate_upload_accepts_supported_type() -> None:
    filename, mime_type = validate_upload("report.pdf", "application/pdf", settings)
    assert filename == "report.pdf"
    assert mime_type == "application/pdf"


def test_validate_upload_resolves_generic_content_type() -> None:
    filename, mime_type = validate_upload("notes.md", "application/octet-stream", settings)
    assert filename == "notes.md"
    assert mime_type == "text/markdown"


def test_validate_upload_rejects_unknown_extension() -> None:
    with pytest.raises(HTTPException) as excinfo:
        validate_upload("malware.exe", "application/octet-stream", settings)
    assert excinfo.value.status_code == 415


def test_validate_upload_rejects_mismatched_content_type() -> None:
    with pytest.raises(HTTPException) as excinfo:
        validate_upload("report.pdf", "image/png", settings)
    assert excinfo.value.status_code == 415


def test_validate_upload_rejects_missing_filename() -> None:
    with pytest.raises(HTTPException) as excinfo:
        validate_upload(None, "application/pdf", settings)
    assert excinfo.value.status_code == 400


def test_validate_upload_is_case_insensitive_for_extension() -> None:
    filename, mime_type = validate_upload("REPORT.PDF", "application/pdf", settings)
    assert filename == "REPORT.PDF"
    assert mime_type == "application/pdf"
