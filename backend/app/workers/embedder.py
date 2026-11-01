"""Embedding providers.

The deterministic provider keeps the full pipeline runnable without external
API keys (local development, CI, demos). Production deployments should set
``EMBEDDING_PROVIDER=openai``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache
from typing import Any, Protocol

from app.core.config import get_settings

logger = logging.getLogger(__name__)

OPENAI_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingProviderError(RuntimeError):
    """Permanent embedding failure (configuration, auth, invalid input)."""


class EmbeddingRequestError(RuntimeError):
    """Transient embedding failure (network, rate limit, provider outage)."""


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicEmbeddingProvider:
    """Hash-based pseudo-embeddings for offline development and tests.

    Vectors are deterministic, unit-normalised and roughly lexical: identical
    tokens land in the same dimensions, so similarity search still behaves
    sensibly without calling an external model.
    """

    model_name = "deterministic-hash-v1"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            for offset in (0, 4, 8):
                index = int.from_bytes(digest[offset : offset + 4], "big") % self.dimension
                sign = 1.0 if digest[offset + 3] & 1 else -1.0
                vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        dimension: int,
        base_url: str | None = None,
        batch_size: int = 64,
        client: Any | None = None,
    ) -> None:
        self.model_name = model
        self.dimension = dimension
        self._batch_size = batch_size
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url or None,
                max_retries=3,
                timeout=60.0,
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
            except Exception as exc:
                raise _translate_openai_error(exc) from exc

            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise EmbeddingProviderError(
                    "Embedding provider returned an unexpected number of vectors"
                )
            vectors.extend(list(item.embedding) for item in data)

        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingProviderError(
                    f"Embedding model {self.model_name!r} returned "
                    f"{len(vector)}-dimensional vectors, expected {self.dimension}. "
                    "Update EMBEDDING_DIMENSION and migrate the schema."
                )

        return [[float(value) for value in vector] for vector in vectors]


def _translate_openai_error(exc: Exception) -> Exception:
    try:
        import openai
    except Exception:  # pragma: no cover - openai is a hard dependency
        return EmbeddingRequestError(str(exc))

    permanent = (
        openai.AuthenticationError,
        openai.BadRequestError,
        openai.NotFoundError,
        openai.PermissionDeniedError,
    )
    if isinstance(exc, permanent):
        return EmbeddingProviderError(str(exc))
    if isinstance(exc, openai.OpenAIError):
        return EmbeddingRequestError(str(exc))
    return EmbeddingRequestError(str(exc))


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()

    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise EmbeddingProviderError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        dimension = OPENAI_MODEL_DIMENSIONS.get(
            settings.embedding_model, settings.embedding_dimension
        )
        if dimension != settings.embedding_dimension:
            raise EmbeddingProviderError(
                f"EMBEDDING_MODEL={settings.embedding_model!r} produces "
                f"{dimension}-dimensional vectors but EMBEDDING_DIMENSION is "
                f"{settings.embedding_dimension}. Change the setting and add a "
                "migration for the chunks.embedding column."
            )
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimension=dimension,
            base_url=settings.openai_base_url,
            batch_size=settings.embedding_batch_size,
        )

    return DeterministicEmbeddingProvider(settings.embedding_dimension)
