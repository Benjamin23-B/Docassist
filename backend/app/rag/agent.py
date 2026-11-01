"""Agentic tool planning and execution.

The deterministic planner maps question intents to a small, auditable tool plan.
``AGENT_PROVIDER=openai`` lets a chat model propose the same JSON plan; every
call is validated against the session scope before it runs, and any planning
failure falls back to the deterministic plan.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.rag.reranker import rerank_chunks
from app.rag.retriever import RetrievedChunk
from app.rag.tools import (
    SUPPORTED_TOOLS,
    ToolCall,
    compare_documents,
    get_page,
    parse_uuid_ids,
    search_documents,
    summarize_documents,
)

logger = logging.getLogger(__name__)

_SUMMARY_RE = re.compile(r"\b(summar(?:y|ise|ize)|overview|tl;?dr|key points|main points)\b")
_COMPARE_RE = re.compile(r"\b(compare|contrast|difference|differ|versus|vs)\b")
_PAGE_RE = re.compile(r"\bpage\s+(\d{1,4})\b")

TOOL_SCHEMA = [
    {
        "name": "search_documents",
        "description": SUPPORTED_TOOLS["search_documents"],
        "arguments": {"query": "string", "document_ids": ["uuid"]},
    },
    {
        "name": "summarize_document",
        "description": SUPPORTED_TOOLS["summarize_document"],
        "arguments": {"document_ids": ["uuid"], "query": "string"},
    },
    {
        "name": "compare_documents",
        "description": SUPPORTED_TOOLS["compare_documents"],
        "arguments": {"document_ids": ["uuid"], "query": "string"},
    },
    {
        "name": "get_page",
        "description": SUPPORTED_TOOLS["get_page"],
        "arguments": {"document_id": "uuid", "page_number": 1},
    },
]


@dataclass(frozen=True, slots=True)
class PlannedTool:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentResult:
    chunks: list[RetrievedChunk]
    tool_calls: list[ToolCall]


class AgentPlanner(Protocol):
    name: str

    def plan(
        self,
        *,
        question: str,
        document_ids: list[uuid.UUID],
        max_calls: int,
    ) -> list[PlannedTool]: ...


class DeterministicAgentPlanner:
    name = "deterministic"

    def plan(
        self,
        *,
        question: str,
        document_ids: list[uuid.UUID],
        max_calls: int,
    ) -> list[PlannedTool]:
        scope = [str(document_id) for document_id in document_ids]
        plans = [
            PlannedTool(
                name="search_documents",
                arguments={"query": question, "document_ids": scope},
            )
        ]
        if max_calls <= 1:
            return plans

        lowered = question.lower()

        if _COMPARE_RE.search(lowered) and len(document_ids) >= 2:
            plans.append(
                PlannedTool(
                    name="compare_documents",
                    arguments={"document_ids": scope, "query": question},
                )
            )
        elif _SUMMARY_RE.search(lowered) and document_ids:
            plans.append(
                PlannedTool(
                    name="summarize_document",
                    arguments={"document_ids": scope, "query": question},
                )
            )

        page_match = _PAGE_RE.search(lowered)
        if page_match and len(document_ids) == 1:
            plans.append(
                PlannedTool(
                    name="get_page",
                    arguments={
                        "document_id": scope[0],
                        "page_number": int(page_match.group(1)),
                    },
                )
            )

        return plans[:max_calls]


class OpenAIAgentPlanner:
    """Ask a chat model to propose a validated JSON tool plan."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            max_retries=1,
            timeout=30.0,
        )
        self._fallback = DeterministicAgentPlanner()

    def plan(
        self,
        *,
        question: str,
        document_ids: list[uuid.UUID],
        max_calls: int,
    ) -> list[PlannedTool]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                max_tokens=400,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Plan document-answering tool calls. Always include "
                            "search_documents first. Use summarize_document for "
                            "summary/overview requests, compare_documents when "
                            "comparing two or more scoped documents, and get_page "
                            "only when the user names a page. Use only document ids "
                            "from the scope. Reply with JSON only: "
                            '{"tool_calls":[{"name":"search_documents",'
                            '"arguments":{"query":"...","document_ids":["..."]}}]}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Scoped document ids: {[str(value) for value in document_ids]}\n"
                            f"Available tools: {json.dumps(TOOL_SCHEMA)}\n"
                            f"Maximum calls: {max_calls}"
                        ),
                    },
                ],
            )
            content = response.choices[0].message.content if response.choices else None
            plans = self._parse_plan(content, max_calls=max_calls)
        except Exception:
            logger.warning("Agent planning failed; using deterministic plan", exc_info=True)
            return self._fallback.plan(
                question=question,
                document_ids=document_ids,
                max_calls=max_calls,
            )

        if not plans:
            return self._fallback.plan(
                question=question,
                document_ids=document_ids,
                max_calls=max_calls,
            )
        return plans

    def _parse_plan(self, content: str | None, *, max_calls: int) -> list[PlannedTool]:
        if not content:
            return []

        raw = content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        calls = payload.get("tool_calls") or payload.get("tools") or []
        if not isinstance(calls, list):
            return []

        plans: list[PlannedTool] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            arguments = call.get("arguments") or {}
            if name not in SUPPORTED_TOOLS or not isinstance(arguments, dict):
                continue
            plans.append(PlannedTool(name=str(name), arguments=arguments))
            if len(plans) >= max_calls:
                break
        return plans


def run_agent(
    db: Session,
    *,
    user_id: uuid.UUID,
    question: str,
    document_ids: list[uuid.UUID],
    candidate_k: int,
    top_k: int,
) -> AgentResult:
    """Plan, execute and merge tool results into grounded passages."""
    settings = get_settings()
    planner = get_agent_planner()
    plans = planner.plan(
        question=question,
        document_ids=document_ids,
        max_calls=settings.agent_max_tool_calls,
    )
    plans = _ensure_search(plans, question=question, document_ids=document_ids)
    plans = plans[: settings.agent_max_tool_calls]

    chunks: list[RetrievedChunk] = []
    tool_calls: list[ToolCall] = []

    for plan in plans:
        try:
            result = _execute_tool(
                db,
                user_id=user_id,
                plan=plan,
                document_ids=document_ids,
                candidate_k=candidate_k,
                question=question,
            )
        except Exception as exc:
            if plan.name == "search_documents":
                # Search is the grounding step: keep the API's existing
                # embedding/DB error mapping rather than answering without it.
                raise
            logger.exception("Agent tool %s failed", plan.name)
            tool_calls.append(
                ToolCall(
                    name=plan.name,
                    arguments=plan.arguments,
                    status="error",
                    detail=str(exc)[:300],
                )
            )
            continue

        tool_calls.append(
            ToolCall(
                name=plan.name,
                arguments=plan.arguments,
                matches=len(result),
            )
        )
        chunks.extend(result)

    deduped: dict[uuid.UUID, RetrievedChunk] = {}
    for chunk in chunks:
        deduped.setdefault(chunk.chunk_id, chunk)

    final = rerank_chunks(
        list(deduped.values()),
        query=question,
        top_k=top_k,
    )
    return AgentResult(chunks=final, tool_calls=tool_calls)


def _ensure_search(
    plans: list[PlannedTool],
    *,
    question: str,
    document_ids: list[uuid.UUID],
) -> list[PlannedTool]:
    if any(plan.name == "search_documents" for plan in plans):
        return plans
    return [
        PlannedTool(
            name="search_documents",
            arguments={
                "query": question,
                "document_ids": [str(document_id) for document_id in document_ids],
            },
        ),
        *plans,
    ]


def _execute_tool(
    db: Session,
    *,
    user_id: uuid.UUID,
    plan: PlannedTool,
    document_ids: list[uuid.UUID],
    candidate_k: int,
    question: str,
) -> list[RetrievedChunk]:
    arguments = plan.arguments
    requested_scope = _resolve_scope(arguments.get("document_ids"), document_ids)

    if plan.name == "search_documents":
        return search_documents(
            db,
            user_id=user_id,
            query=_query_argument(arguments, question),
            document_ids=requested_scope,
            candidate_k=candidate_k,
        )
    if plan.name == "summarize_document":
        return summarize_documents(
            db,
            user_id=user_id,
            document_ids=requested_scope,
            query=_query_argument(arguments, question),
            candidate_k=candidate_k,
        )
    if plan.name == "compare_documents":
        return compare_documents(
            db,
            user_id=user_id,
            document_ids=requested_scope,
            query=_query_argument(arguments, question),
            candidate_k=candidate_k,
        )
    if plan.name == "get_page":
        document_id = _resolve_document(arguments.get("document_id"), document_ids)
        try:
            page_number = int(arguments.get("page_number", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("page_number must be an integer") from exc
        return get_page(
            db,
            user_id=user_id,
            document_id=document_id,
            page_number=page_number,
        )
    raise ValueError(f"Unsupported tool: {plan.name}")


def _query_argument(arguments: dict[str, Any], fallback: str) -> str:
    query = arguments.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()
    return fallback


def _resolve_scope(requested: Any, session_scope: list[uuid.UUID]) -> list[uuid.UUID]:
    if requested is None:
        return list(session_scope)
    ids = parse_uuid_ids(requested)
    if session_scope and any(document_id not in session_scope for document_id in ids):
        raise ValueError("Requested document is outside the session scope")
    return ids


def _resolve_document(requested: Any, session_scope: list[uuid.UUID]) -> uuid.UUID:
    if requested is None:
        if len(session_scope) == 1:
            return session_scope[0]
        raise ValueError("document_id is required")
    document_id = uuid.UUID(str(requested))
    if session_scope and document_id not in session_scope:
        raise ValueError("Requested document is outside the session scope")
    return document_id


@lru_cache(maxsize=1)
def get_agent_planner() -> DeterministicAgentPlanner | OpenAIAgentPlanner:
    settings = get_settings()
    if settings.agent_provider == "openai" and settings.openai_api_key:
        return OpenAIAgentPlanner(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            base_url=settings.openai_base_url,
        )
    return DeterministicAgentPlanner()
