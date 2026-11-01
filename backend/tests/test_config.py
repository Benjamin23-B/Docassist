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


def test_agent_and_rate_limit_defaults_are_offline_friendly() -> None:
    settings = _settings()

    assert settings.agent_enabled is True
    assert settings.agent_provider == "deterministic"
    assert settings.agent_max_tool_calls == 4
    assert settings.rate_limit_enabled is True
    assert settings.rate_limit_requests > 0
    assert settings.chat_rate_limit_requests > 0


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
        ({"agent_max_tool_calls": 0}, "AGENT_MAX_TOOL_CALLS"),
        ({"rate_limit_requests": 0}, "RATE_LIMIT_REQUESTS"),
        ({"rate_limit_window_seconds": 0}, "RATE_LIMIT_WINDOW_SECONDS"),
        ({"chat_rate_limit_requests": 0}, "CHAT_RATE_LIMIT_REQUESTS"),
        ({"chat_rate_limit_window_seconds": 0}, "CHAT_RATE_LIMIT_WINDOW_SECONDS"),
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


def test_openai_agent_planning_requires_a_key() -> None:
    settings = _settings(agent_provider="openai", openai_api_key=None)

    with pytest.raises(RuntimeError, match="AGENT_PROVIDER=openai"):
        settings.validate_runtime()
