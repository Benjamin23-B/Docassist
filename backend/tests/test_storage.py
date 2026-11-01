from urllib.parse import unquote

from app.core.config import Settings
from app.services.storage import StorageService


def _storage(**overrides: object) -> StorageService:
    values: dict[str, object] = {
        "s3_endpoint_url": "http://minio:9000",
        "s3_public_endpoint_url": "http://localhost:9000",
        "s3_access_key_id": "test-key",
        "s3_secret_access_key": "test-secret",
    }
    values.update(overrides)
    return StorageService(Settings(**values))


def test_presigned_urls_use_public_endpoint() -> None:
    url = _storage().presigned_download_url("users/u/documents/d/f.pdf", "f.pdf")
    assert url.startswith("http://localhost:9000/rag-documents/")
    assert "minio:9000" not in url


def test_presigned_urls_fall_back_to_internal_endpoint() -> None:
    storage = _storage(s3_public_endpoint_url=None)
    url = storage.presigned_download_url("users/u/documents/d/f.pdf", "f.pdf")
    assert url.startswith("http://minio:9000/rag-documents/")


def test_presigned_url_encodes_filename_once() -> None:
    url = _storage().presigned_download_url("k", "my report (final).pdf")
    assert "%2520" not in url
    assert "my report (final).pdf" in unquote(url)
