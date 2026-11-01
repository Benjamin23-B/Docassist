import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_advanced_retrieval_defaults_are_offline_friendly() -> None:
    settings = _settings()

    assert settings.query_rewrite_enabled is True
    assert settings.query_rewrite_provider == "deterministic"
    assert settings.hybrid_search_enabled is True
    assert settings.reranker_provider == "deterministic"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hybrid_candidate_multiplier": 0}, "HYBRID_CANDIDATE_MULTIPLIER"),
        ({"hybrid_rrf_k": 0}, "HYBRID_RRF_K"),
        ({"hybrid_vector_weight": -1}, "weights"),
        (
            {"hybrid_vector_weight": 0, "hybrid_keyword_weight": 0},
            "At least one hybrid retrieval weight",
        ),
        (
            {"retrieval_top_k": 10, "rerank_candidate_k": 5},
            "RERANK_CANDIDATE_K",
        ),
    ],
)
def test_invalid_hybrid_settings_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _settings(**overrides)


def test_openai_query_rewriting_requires_a_key() -> None:
    settings = _settings(query_rewrite_provider="openai", openai_api_key=None)

    with pytest.raises(RuntimeError, match="QUERY_REWRITE_PROVIDER=openai"):
        settings.validate_runtime()


def test_cohere_reranking_requires_a_key() -> None:
    settings = _settings(reranker_provider="cohere", cohere_api_key=None)

    with pytest.raises(RuntimeError, match="RERANKER_PROVIDER=cohere"):
        settings.validate_runtime()
