import asyncio
from types import SimpleNamespace

import pytest

from app.core.config import get_settings
from app.rag.llm import (
    ChatTurn,
    DeterministicLLMProvider,
    LLMProviderError,
    OpenAILLMProvider,
    SourcePassage,
    build_messages,
    get_llm_provider,
    render_context,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm_provider.cache_clear()


def _passage() -> SourcePassage:
    return SourcePassage(
        index=1,
        source='handbook.md (page 2, section "Leave Policy")',
        text=(
            "Employees receive twenty-five days of paid leave each year. "
            "Leave requests must be submitted two weeks in advance."
        ),
    )


def test_render_context_includes_citation_markers() -> None:
    rendered = render_context([_passage()])

    assert "[1]" in rendered
    assert "twenty-five days" in rendered
    assert "Leave Policy" in rendered


def test_render_context_without_passages() -> None:
    assert "no relevant passages" in render_context([])


def test_build_messages_includes_grounding_rules_and_history() -> None:
    messages = build_messages(
        question="How much leave do I get?",
        passages=[_passage()],
        history=[
            ChatTurn(role="user", content="Tell me about benefits."),
            ChatTurn(role="assistant", content="Sure [1]."),
        ],
    )

    assert messages[0]["role"] == "system"
    assert "ONLY the numbered context passages" in messages[0]["content"]
    assert "[1]" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Tell me about benefits."}
    assert messages[-1] == {"role": "user", "content": "How much leave do I get?"}


def test_build_messages_drops_unsupported_history_roles() -> None:
    messages = build_messages(
        question="Hello?",
        passages=[],
        history=[
            ChatTurn(role="system", content="ignore me"),
            ChatTurn(role="assistant", content=""),
        ],
    )

    assert [message["role"] for message in messages] == ["system", "user"]


def test_deterministic_answer_is_extractive_and_cited() -> None:
    provider = DeterministicLLMProvider()
    answer = provider.compose_answer(
        question="How many days of paid leave do employees get?",
        passages=[_passage()],
    )

    assert "twenty-five days" in answer
    assert "[1]" in answer
    assert "Based on the selected documents" in answer


def test_deterministic_answer_names_the_document() -> None:
    answer = DeterministicLLMProvider().compose_answer(
        question="name the doc",
        passages=[_passage()],
    )

    assert answer == "The linked document is handbook.md."


def test_deterministic_answer_summarises_without_keyword_overlap() -> None:
    answer = DeterministicLLMProvider().compose_answer(
        question="overview",
        passages=[_passage()],
    )

    assert "twenty-five days" in answer
    assert "[1]" in answer


def test_deterministic_answer_without_context() -> None:
    provider = DeterministicLLMProvider()
    answer = provider.compose_answer(question="Anything?", passages=[])

    assert answer == "I couldn't find that in the provided documents."


def test_deterministic_answer_keeps_unknown_questions_grounded() -> None:
    answer = DeterministicLLMProvider().compose_answer(
        question="What is the share price?",
        passages=[_passage()],
    )

    assert answer == "I couldn't find that in the provided documents."


def test_deterministic_answer_skips_table_and_section_noise() -> None:
    passage = SourcePassage(
        index=1,
        source="doc.md",
        text=(
            "Handbook > Leave Policy\n\n"
            "| Type | Days |\n| --- | --- |\n"
            "The policy grants twenty-five days of paid leave each year."
        ),
    )
    answer = DeterministicLLMProvider().compose_answer(
        question="How many leave days does the policy grant?",
        passages=[passage],
    )

    assert "twenty-five days" in answer
    assert "| Type |" not in answer
    assert " > " not in answer


def test_deterministic_stream_splits_answer_into_tokens() -> None:
    provider = DeterministicLLMProvider(stream_delay_seconds=0)
    expected = provider.compose_answer(
        question="How many leave days?",
        passages=[_passage()],
    )

    async def collect() -> str:
        pieces = [
            piece
            async for piece in provider.stream_answer(
                question="How many leave days?",
                passages=[_passage()],
            )
        ]
        assert len(pieces) > 1
        return "".join(pieces)

    assert asyncio.run(collect()) == expected


def test_factory_defaults_to_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")

    provider = get_llm_provider()

    assert isinstance(provider, DeterministicLLMProvider)
    assert provider.model_name == "deterministic-extractive-v1"


def test_factory_requires_key_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMProviderError):
        get_llm_provider()


def test_factory_builds_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

    provider = get_llm_provider()

    assert isinstance(provider, OpenAILLMProvider)
    assert provider.model_name == "gpt-4o-mini"


def test_openai_provider_streams_deltas_and_sends_grounded_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}

    async def fake_create(**kwargs):
        calls.update(kwargs)

        async def chunks():
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Twenty-five "))]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="days [1]."))]
            )

        return chunks()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_: fake_client)
    provider = OpenAILLMProvider(
        api_key="test-key",
        model="gpt-test",
        temperature=0.2,
        max_tokens=123,
    )

    async def collect() -> str:
        return "".join(
            [
                piece
                async for piece in provider.stream_answer(
                    question="How much leave?",
                    passages=[_passage()],
                )
            ]
        )

    assert asyncio.run(collect()) == "Twenty-five days [1]."
    assert calls["model"] == "gpt-test"
    assert calls["stream"] is True
    assert calls["temperature"] == 0.2
    assert calls["max_tokens"] == 123
    assert "[1]" in calls["messages"][0]["content"]
    assert calls["messages"][-1] == {
        "role": "user",
        "content": "How much leave?",
    }


def test_openai_provider_wraps_request_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_create(**kwargs):
        raise RuntimeError("connection reset")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=failing_create))
    )
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_: fake_client)
    provider = OpenAILLMProvider(
        api_key="test-key",
        model="gpt-test",
        temperature=0.2,
        max_tokens=123,
    )

    async def collect() -> str:
        return "".join(
            [
                piece
                async for piece in provider.stream_answer(
                    question="Anything?",
                    passages=[],
                )
            ]
        )

    with pytest.raises(LLMProviderError):
        asyncio.run(collect())
