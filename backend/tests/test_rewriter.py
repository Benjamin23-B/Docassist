from app.rag.llm import ChatTurn
from app.rag.rewriter import (
    DeterministicQueryRewriter,
    is_follow_up,
    rewrite_query,
)

HISTORY = [
    ChatTurn(role="user", content="How many days of leave do employees get?"),
    ChatTurn(role="assistant", content="Employees receive twenty-five days [1]."),
]


def test_follow_up_detection_uses_pronouns_phrases_and_context() -> None:
    assert not is_follow_up("How many days is it?", None)
    assert is_follow_up("How many days is it?", "How much leave is offered?")
    assert is_follow_up("What about parental leave?", "What is the leave policy?")
    assert is_follow_up("And the deadline?", "When is enrolment due?")
    assert not is_follow_up(
        "How many days do employees get?",
        "What is the leave policy?",
    )


def test_deterministic_rewriter_resolves_follow_ups() -> None:
    rewritten = DeterministicQueryRewriter().rewrite("Is it paid?", HISTORY)

    assert rewritten.startswith("How many days of leave do employees get?")
    assert rewritten.endswith("Is it paid?")


def test_deterministic_rewriter_leaves_standalone_questions_alone() -> None:
    question = "What is the security badge policy?"

    assert DeterministicQueryRewriter().rewrite(question, HISTORY) == question


def test_deterministic_rewriter_needs_history() -> None:
    assert DeterministicQueryRewriter().rewrite("Is it paid?", []) == "Is it paid?"


def test_rewrite_query_wrapper_uses_deterministic_default() -> None:
    rewritten = rewrite_query("Is it paid?", HISTORY)

    assert "How many days of leave" in rewritten
    assert "Is it paid?" in rewritten
