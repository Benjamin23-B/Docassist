from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_JWT_SECRET = "dev-only-secret-please-change-me-0123456789abcdef"
INSECURE_JWT_SECRETS = {
    DEFAULT_JWT_SECRET,
    "change-me-in-production",
    "secret",
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    project_name: str = "RAG Document Q&A Platform"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    jwt_secret: str = DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    s3_endpoint_url: str | None = "http://localhost:9000"
    # Endpoint embedded in presigned URLs. In Docker the API talks to
    # `http://minio:9000`, but browsers need a host-reachable address.
    s3_public_endpoint_url: str | None = None
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: str = "minioadmin"
    s3_bucket: str = "rag-documents"
    s3_region: str = "us-east-1"

    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [
        ".pdf",
        ".md",
        ".markdown",
        ".txt",
        ".docx",
        ".html",
        ".htm",
    ]

    embedding_dimension: int = 1536

    @field_validator("allowed_extensions")
    @classmethod
    def _normalise_extensions(cls, value: list[str]) -> list[str]:
        normalised: list[str] = []
        for extension in value:
            extension = extension.strip().lower()
            if not extension:
                continue
            if not extension.startswith("."):
                extension = f".{extension}"
            normalised.append(extension)
        return normalised

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def validate_runtime(self) -> None:
        """Fail fast on unsafe configuration in production."""
        if self.environment.lower() != "production":
            return
        if self.jwt_secret in INSECURE_JWT_SECRETS or len(self.jwt_secret) < 32:
            raise RuntimeError(
                "JWT_SECRET must be a unique value of at least 32 characters "
                "before running in production."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
