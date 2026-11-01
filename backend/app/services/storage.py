"""S3-compatible object storage (MinIO locally, AWS S3 in production)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import BinaryIO

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Raised when an object storage operation fails."""


class StorageService:
    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.s3_bucket
        self._region = settings.s3_region
        client_kwargs = {
            "aws_access_key_id": settings.s3_access_key_id,
            "aws_secret_access_key": settings.s3_secret_access_key,
            "region_name": settings.s3_region,
            "config": Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=5,
                read_timeout=60,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        }
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url or None,
            **client_kwargs,
        )
        # Signed URLs are generated against a client-facing endpoint so the
        # browser/desktop client can actually reach the object.
        self._presign_client = boto3.client(
            "s3",
            endpoint_url=(settings.s3_public_endpoint_url or settings.s3_endpoint_url or None),
            **client_kwargs,
        )

    @property
    def bucket(self) -> str:
        return self._bucket

    def ensure_bucket(self) -> None:
        """Create the configured bucket when it does not exist yet."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise StorageError(f"Unable to access bucket {self._bucket!r}") from exc

        create_kwargs: dict = {"Bucket": self._bucket}
        if self._region and self._region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self._region}
        try:
            self._client.create_bucket(**create_kwargs)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise StorageError(f"Unable to create bucket {self._bucket!r}") from exc

    def upload_fileobj(self, fileobj: BinaryIO, key: str, content_type: str) -> None:
        try:
            self._client.upload_fileobj(
                fileobj,
                self._bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"Failed to upload object {key!r}") from exc

    def delete_object(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"Failed to delete object {key!r}") from exc

    def presigned_download_url(
        self,
        key: str,
        filename: str,
        expires_in: int = 3600,
    ) -> str:
        disposition = f"attachment; filename*=UTF-8''{filename}"
        try:
            return self._presign_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self._bucket,
                    "Key": key,
                    "ResponseContentDisposition": disposition,
                },
                ExpiresIn=expires_in,
            )
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"Failed to create download URL for {key!r}") from exc


@lru_cache
def get_storage() -> StorageService:
    return StorageService(get_settings())
