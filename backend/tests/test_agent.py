import uuid

import pytest

from app.rag.agent import (
    DeterministicAgentPlanner,
    OpenAIAgentPlanner,
    PlannedTool,
    _resolve_document,
    _resolve_scope,
    run_agent,
)
from app.rag.tools import parse_uuid_ids


def _ids(count: int) -> list[uuid.UUID]:
    return [uuid.uuid4() for _ in range(count)]


def test_planner_always_searches_first() -> None:
    plans = DeterministicAgentPlanner().plan(
        question="What is the leave policy?",
        document_ids=_ids(1),
        max_calls=4,
    )

    assert [plan.name for plan in plans] == ["search_documents"]
    assert plans[0].arguments["query"] == "What is the leave policy?"


def test_planner_adds_summarize_for_summary_intent() -> None:
    document_ids = _ids(2)

    plans = DeterministicAgentPlanner().plan(
        question="Give me an overview of these documents",
        document_ids=document_ids,
        max_calls=4,
    )

    assert [plan.name for plan in plans] == ["search_documents", "summarize_document"]
    assert plans[1].arguments["document_ids"] == [str(value) for value in document_ids]


def test_planner_prefers_compare_for_comparison_intent() -> None:
    plans = DeterministicAgentPlanner().plan(
        question="Compare the leave policy across both documents",
        document_ids=_ids(2),
        max_calls=4,
    )

    assert [plan.name for plan in plans] == ["search_documents", "compare_documents"]


def test_planner_adds_get_page_for_single_scoped_document() -> None:
    document_ids = _ids(1)

    plans = DeterministicAgentPlanner().plan(
        question="What does page 12 say about expenses?",
        document_ids=document_ids,
        max_calls=4,
    )

    assert [plan.name for plan in plans] == ["search_documents", "get_page"]
    assert plans[1].arguments == {
        "document_id": str(document_ids[0]),
        "page_number": 12,
    }


def test_planner_respects_the_call_budget() -> None:
    plans = DeterministicAgentPlanner().plan(
        question="Compare page 2 overview",
        document_ids=_ids(2),
        max_calls=1,
    )

    assert [plan.name for plan in plans] == ["search_documents"]


def test_scope_rejects_documents_outside_the_session() -> None:
    allowed = _ids(1)
    outside = _ids(1)

    with pytest.raises(ValueError, match="outside the session scope"):
        _resolve_scope([str(outside[0])], allowed)

    with pytest.raises(ValueError, match="outside the session scope"):
        _resolve_document(str(outside[0]), allowed)


def test_scope_defaults_to_the_session_documents() -> None:
    allowed = _ids(2)

    assert _resolve_scope(None, allowed) == allowed
    assert _resolve_document(None, allowed[:1]) == allowed[0]


def test_parse_uuid_ids_rejects_non_lists() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        parse_uuid_ids("not-a-list")


def test_openai_plan_parser_accepts_fenced_json() -> None:
    planner = object.__new__(OpenAIAgentPlanner)
    plans = planner._parse_plan(
        '```json\n{"tool_calls": [{"name": "search_documents", '
        '"arguments": {"query": "leave"}}]}\n```',
        max_calls=2,
    )

    assert plans == [PlannedTool("search_documents", {"query": "leave"})]


def test_openai_plan_parser_ignores_unknown_tools() -> None:
    planner = object.__new__(OpenAIAgentPlanner)
    plans = planner._parse_plan(
        '{"tool_calls": [{"name": "drop_tables", "arguments": {}}, '
        '{"name": "get_page", "arguments": {"page_number": 2}}]}',
        max_calls=2,
    )

    assert plans == [PlannedTool("get_page", {"page_number": 2})]


def test_run_agent_propagates_search_failures(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("app.rag.agent._execute_tool", fail)

    with pytest.raises(RuntimeError, match="database unavailable"):
        run_agent(
            None,
            user_id=uuid.uuid4(),
            question="What is the leave policy?",
            document_ids=[],
            candidate_k=10,
            top_k=4,
        )


def test_run_agent_records_non_search_tool_errors(monkeypatch) -> None:
    def execute(_db, *, plan, **kwargs):
        if plan.name == "summarize_document":
            raise ValueError("document has no readable chunks")
        return []

    monkeypatch.setattr("app.rag.agent._execute_tool", execute)

    result = run_agent(
        None,
        user_id=uuid.uuid4(),
        question="Summarize the leave policy",
        document_ids=[uuid.uuid4()],
        candidate_k=10,
        top_k=4,
    )

    assert [call.name for call in result.tool_calls] == [
        "search_documents",
        "summarize_document",
    ]
    assert result.tool_calls[0].status == "ok"
    assert result.tool_calls[1].status == "error"
    assert result.tool_calls[1].detail == "document has no readable chunks"
