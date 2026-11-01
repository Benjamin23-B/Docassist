from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
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
    redis_url: str = "redis://localhost:6379/0"

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

    # --- Ingestion / embeddings ---------------------------------------------
    embedding_provider: Literal["deterministic", "openai"] = "deterministic"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 64
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 90
    min_chunk_tokens: int = 120

    ingestion_max_retries: int = 3
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    celery_task_always_eager: bool = False
    dlq_key: str = "rag:ingestion:dlq"

    # --- Retrieval & generation ---------------------------------------------
    retrieval_top_k: int = 6
    chat_history_limit: int = 8

    # --- Advanced retrieval (Phase 5) ---------------------------------------
    query_rewrite_enabled: bool = True
    query_rewrite_provider: Literal["deterministic", "openai"] = "deterministic"
    # Defaults to LLM_MODEL when unset.
    query_rewrite_model: str | None = None
    query_rewrite_history_limit: int = 4

    hybrid_search_enabled: bool = True
    # Each retriever fetches TOP_K * multiplier candidates before fusion.
    hybrid_candidate_multiplier: int = 4
    hybrid_rrf_k: int = 60
    hybrid_vector_weight: float = 1.0
    hybrid_keyword_weight: float = 1.0

    reranker_provider: Literal["none", "deterministic", "cohere"] = "deterministic"
    # Retrieve this many candidates, then keep RETRIEVAL_TOP_K after reranking.
    rerank_candidate_k: int = 20
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-english-v3.0"
    cohere_base_url: str = "https://api.cohere.com/v1"

    llm_provider: Literal["deterministic", "openai"] = "deterministic"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 800
    llm_stream_delay_seconds: float = 0.02

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

    @field_validator(
        "s3_endpoint_url",
        "s3_public_endpoint_url",
        "openai_api_key",
        "openai_base_url",
        "celery_broker_url",
        "celery_result_backend",
        "query_rewrite_model",
        "cohere_api_key",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _validate_ingestion_settings(self) -> "Settings":
        if self.embedding_dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be positive")
        if self.chunk_size_tokens <= 0:
            raise ValueError("CHUNK_SIZE_TOKENS must be positive")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("CHUNK_OVERLAP_TOKENS must not be negative")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.min_chunk_tokens > self.chunk_size_tokens:
            raise ValueError("MIN_CHUNK_TOKENS must not exceed CHUNK_SIZE_TOKENS")
        if self.embedding_batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        if self.retrieval_top_k <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be positive")
        if self.chat_history_limit < 0:
            raise ValueError("CHAT_HISTORY_LIMIT must not be negative")
        if self.query_rewrite_history_limit < 0:
            raise ValueError("QUERY_REWRITE_HISTORY_LIMIT must not be negative")
        if self.hybrid_candidate_multiplier < 1:
            raise ValueError("HYBRID_CANDIDATE_MULTIPLIER must be at least 1")
        if self.hybrid_rrf_k <= 0:
            raise ValueError("HYBRID_RRF_K must be positive")
        if self.hybrid_vector_weight < 0 or self.hybrid_keyword_weight < 0:
            raise ValueError("Hybrid retrieval weights must not be negative")
        if self.hybrid_vector_weight == 0 and self.hybrid_keyword_weight == 0:
            raise ValueError("At least one hybrid retrieval weight must be positive")
        if self.rerank_candidate_k < self.retrieval_top_k:
            raise ValueError("RERANK_CANDIDATE_K must not be smaller than RETRIEVAL_TOP_K")
        if self.llm_max_tokens <= 0:
            raise ValueError("LLM_MAX_TOKENS must be positive")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if self.llm_stream_delay_seconds < 0:
            raise ValueError("LLM_STREAM_DELAY_SECONDS must not be negative")
        return self

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def validate_runtime(self) -> None:
        """Fail fast on unsafe or incomplete configuration."""
        if "openai" in {self.embedding_provider, self.llm_provider} and not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER or "
                "LLM_PROVIDER is set to openai"
            )
        if self.query_rewrite_enabled and self.query_rewrite_provider == "openai":
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required when QUERY_REWRITE_PROVIDER=openai")
        if self.reranker_provider == "cohere" and not self.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is required when RERANKER_PROVIDER=cohere")
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
