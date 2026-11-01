import math
from types import SimpleNamespace

import pytest

from app.core.config import get_settings
from app.workers.embedder import (
    DeterministicEmbeddingProvider,
    EmbeddingProviderError,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    get_settings.cache_clear()
    get_embedding_provider.cache_clear()
    yield
    get_settings.cache_clear()
    get_embedding_provider.cache_clear()


def test_deterministic_provider_is_stable_and_normalised() -> None:
    provider = DeterministicEmbeddingProvider(dimension=64)

    first = provider.embed(["hello world"])[0]
    second = provider.embed(["hello world"])[0]
    different = provider.embed(["different text"])[0]

    assert first == second
    assert first != different
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(value**2 for value in first)), 1.0)


def test_deterministic_provider_handles_empty_text() -> None:
    vector = DeterministicEmbeddingProvider(dimension=8).embed([""])[0]
    assert vector[0] == 1.0
    assert sum(value**2 for value in vector) == 1.0


def test_factory_defaults_to_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "deterministic")

    provider = get_embedding_provider()

    assert isinstance(provider, DeterministicEmbeddingProvider)
    assert provider.dimension == 1536


def test_factory_requires_key_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(EmbeddingProviderError):
        get_embedding_provider()


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append(input)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index)] * 4)
                for index in range(len(input))
            ]
        )


def test_openai_provider_batches_requests() -> None:
    embeddings = _FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    provider = OpenAIEmbeddingProvider(
        api_key=None,
        model="test-model",
        dimension=4,
        batch_size=2,
        client=client,
    )

    vectors = provider.embed(["one", "two", "three"])

    assert [len(call) for call in embeddings.calls] == [2, 1]
    assert len(vectors) == 3
    assert all(len(vector) == 4 for vector in vectors)


def test_openai_provider_rejects_dimension_mismatch() -> None:
    embeddings = _FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    provider = OpenAIEmbeddingProvider(
        api_key=None,
        model="test-model",
        dimension=8,
        batch_size=2,
        client=client,
    )

    with pytest.raises(EmbeddingProviderError):
        provider.embed(["one"])
