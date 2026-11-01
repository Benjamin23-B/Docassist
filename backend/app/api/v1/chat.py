"""Chat sessions and grounded SSE question answering."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, DbSession
from app.core.config import get_settings
from app.db.models import ChatSession, Document, Message, MessageRole
from app.db.session import SessionLocal
from app.rag.llm import (
    ChatTurn,
    LLMProviderError,
    SourcePassage,
    get_llm_provider,
)
from app.rag.reranker import rerank_chunks
from app.rag.retriever import RetrievedChunk, retrieve_chunks
from app.rag.rewriter import rewrite_query
from app.schemas.chat import (
    AskRequest,
    ChatMessageRead,
    ChatSessionCreate,
    ChatSessionDetail,
    ChatSessionListResponse,
    ChatSessionRead,
    ChatSessionUpdate,
    Citation,
)
from app.services.export import export_filename, render_session_markdown
from app.workers.chunker import count_tokens
from app.workers.embedder import EmbeddingProviderError, EmbeddingRequestError

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

DEFAULT_SESSION_TITLE = "New chat"


@router.post(
    "/sessions",
    response_model=ChatSessionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a chat session with optional document scoping",
    responses={400: {"description": "Unknown document ids"}},
)
def create_session(
    payload: ChatSessionCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> ChatSession:
    document_ids = _validate_document_scope(
        db, user_id=current_user.id, document_ids=payload.document_ids
    )
    session = ChatSession(
        user_id=current_user.id,
        title=payload.title or DEFAULT_SESSION_TITLE,
        doc_scope=document_ids,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.patch(
    "/sessions/{session_id}",
    response_model=ChatSessionRead,
    summary="Rename a chat session",
)
def rename_session(
    session_id: uuid.UUID,
    payload: ChatSessionUpdate,
    db: DbSession,
    current_user: CurrentUser,
) -> ChatSession:
    session = _get_owned_session(session_id, db, current_user.id)
    session.title = payload.title
    db.commit()
    db.refresh(session)
    return session


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session with its messages and feedback",
)
def delete_session(
    session_id: uuid.UUID,
    db: DbSession,
    current_user: CurrentUser,
) -> Response:
    session = _get_owned_session(session_id, db, current_user.id)
    db.delete(session)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/sessions",
    response_model=ChatSessionListResponse,
    summary="List the current user's chat sessions",
)
def list_sessions(
    db: DbSession,
    current_user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> ChatSessionListResponse:
    filters = (ChatSession.user_id == current_user.id,)
    total = db.scalar(select(func.count()).select_from(ChatSession).where(*filters)) or 0
    sessions = db.scalars(
        select(ChatSession)
        .where(*filters)
        .order_by(ChatSession.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()

    return ChatSessionListResponse(
        items=[ChatSessionRead.model_validate(session) for session in sessions],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionDetail,
    summary="Get a chat session with its message history",
)
def get_session(
    session_id: uuid.UUID,
    db: DbSession,
    current_user: CurrentUser,
) -> ChatSessionDetail:
    session = _get_owned_session(session_id, db, current_user.id)
    messages = db.scalars(
        select(Message)
        .options(selectinload(Message.feedback))
        .where(Message.session_id == session.id)
        .order_by(Message.created_at, Message.id)
    ).all()

    return ChatSessionDetail(
        id=session.id,
        title=session.title,
        doc_scope=session.doc_scope,
        created_at=session.created_at,
        messages=[ChatMessageRead.model_validate(message) for message in messages],
    )


@router.get(
    "/sessions/{session_id}/export",
    summary="Download a chat session as Markdown",
    response_class=Response,
    responses={
        200: {
            "content": {"text/markdown": {}},
            "description": "Markdown transcript with citations and feedback.",
        }
    },
)
def export_session(
    session_id: uuid.UUID,
    db: DbSession,
    current_user: CurrentUser,
) -> Response:
    session = _get_owned_session(session_id, db, current_user.id)
    messages = db.scalars(
        select(Message)
        .options(selectinload(Message.feedback))
        .where(Message.session_id == session.id)
        .order_by(Message.created_at, Message.id)
    ).all()
    markdown = render_session_markdown(
        session_id=session.id,
        title=session.title,
        created_at=session.created_at,
        messages=messages,
    )
    filename = export_filename(session.title, session_id=session.id)
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/sessions/{session_id}/ask",
    summary="Ask a question and stream the grounded answer over SSE",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": ("SSE stream with `meta`, `token`, `done` and `error` events."),
        }
    },
)
def ask_question(
    session_id: uuid.UUID,
    payload: AskRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> StreamingResponse:
    session = _get_owned_session(session_id, db, current_user.id)
    session_id_value = session.id
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question must not be empty",
        )

    history = _load_history(db, session_id_value)
    search_query = rewrite_query(question, history)
    if search_query != " ".join(question.split()):
        logger.info(
            "Rewrote follow-up for session %s: %r -> %r",
            session_id_value,
            question,
            search_query,
        )

    reranking = settings.reranker_provider != "none"
    candidate_k = settings.rerank_candidate_k if reranking else settings.retrieval_top_k

    try:
        chunks = retrieve_chunks(
            db,
            user_id=current_user.id,
            query=search_query,
            doc_scope=session.doc_scope,
            top_k=candidate_k,
        )
    except EmbeddingProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding provider is not configured: {exc}",
        ) from exc
    except EmbeddingRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The embedding provider is unavailable. Please retry.",
        ) from exc

    chunks = rerank_chunks(
        chunks,
        query=search_query,
        top_k=settings.retrieval_top_k,
    )

    try:
        llm = get_llm_provider()
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Language model is not configured: {exc}",
        ) from exc

    passages, citations = _build_context(chunks)

    db.add(
        Message(
            session_id=session_id_value,
            role=MessageRole.USER.value,
            content=question,
        )
    )
    if session.title == DEFAULT_SESSION_TITLE:
        session.title = " ".join(question.split())[:80]
    db.commit()

    assistant_message_id = uuid.uuid4()
    citation_payload = [citation.model_dump(mode="json") for citation in citations]

    async def event_stream() -> AsyncIterator[str]:
        started = time.perf_counter()
        answer_parts: list[str] = []

        yield _sse(
            "meta",
            {
                "session_id": str(session_id_value),
                "message_id": str(assistant_message_id),
                "model": llm.model_name,
                "citations": citation_payload,
                "query": search_query,
                "retrieval": {
                    "rewritten": search_query != " ".join(question.split()),
                    "hybrid": settings.hybrid_search_enabled,
                    "reranked": reranking,
                    "matches": len(chunks),
                },
            },
        )

        try:
            async for token in llm.stream_answer(
                question=question,
                passages=passages,
                history=history,
            ):
                answer_parts.append(token)
                yield _sse("token", {"text": token})
        except Exception as exc:
            logger.exception("Chat generation failed for session %s", session_id_value)
            partial = "".join(answer_parts).strip()
            if partial:
                _persist_assistant_message(
                    message_id=assistant_message_id,
                    session_id=session_id_value,
                    content=partial,
                    citations=citations,
                    latency_ms=_elapsed_ms(started),
                )
            detail = (
                str(exc)
                if settings.environment.lower() != "production"
                else "The assistant could not complete the answer."
            )
            yield _sse(
                "error",
                {
                    "detail": detail,
                    "message_id": str(assistant_message_id),
                },
            )
            return

        answer = "".join(answer_parts).strip()
        latency_ms = _elapsed_ms(started)
        tokens_used = count_tokens(answer)
        _persist_assistant_message(
            message_id=assistant_message_id,
            session_id=session_id_value,
            content=answer,
            citations=citations,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )
        yield _sse(
            "done",
            {
                "message_id": str(assistant_message_id),
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _validate_document_scope(
    db: DbSession,
    *,
    user_id: uuid.UUID,
    document_ids: list[uuid.UUID],
) -> list[uuid.UUID]:
    unique_ids = list(dict.fromkeys(document_ids))
    if not unique_ids:
        return []

    owned = set(
        db.scalars(
            select(Document.id).where(
                Document.user_id == user_id,
                Document.id.in_(unique_ids),
            )
        ).all()
    )
    unknown = [str(document_id) for document_id in unique_ids if document_id not in owned]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown document ids: {', '.join(unknown)}",
        )
    return unique_ids


def _get_owned_session(session_id: uuid.UUID, db: DbSession, user_id: uuid.UUID) -> ChatSession:
    session = db.scalar(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )
    return session


def _load_history(db: DbSession, session_id: uuid.UUID) -> list[ChatTurn]:
    if settings.chat_history_limit <= 0:
        return []

    rows = db.scalars(
        select(Message)
        .where(
            Message.session_id == session_id,
            Message.role.in_([MessageRole.USER.value, MessageRole.ASSISTANT.value]),
        )
        .order_by(Message.created_at.desc())
        .limit(settings.chat_history_limit)
    ).all()
    return [ChatTurn(role=row.role, content=row.content) for row in reversed(rows)]


def _build_context(
    chunks: list[RetrievedChunk],
) -> tuple[list[SourcePassage], list[Citation]]:
    passages: list[SourcePassage] = []
    citations: list[Citation] = []

    for index, chunk in enumerate(chunks, start=1):
        source = chunk.filename
        details: list[str] = []
        if chunk.page_number is not None:
            details.append(f"page {chunk.page_number}")
        if chunk.section:
            details.append(f'section "{chunk.section}"')
        if details:
            source = f"{source} ({', '.join(details)})"

        passages.append(SourcePassage(index=index, source=source, text=chunk.content))
        citations.append(
            Citation(
                index=index,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                section=chunk.section,
                score=chunk.score,
                snippet=chunk.snippet,
            )
        )

    return passages, citations


def _persist_assistant_message(
    *,
    message_id: uuid.UUID,
    session_id: uuid.UUID,
    content: str,
    citations: list[Citation],
    latency_ms: int,
    tokens_used: int | None = None,
) -> None:
    if not content:
        return

    try:
        with SessionLocal() as stream_db:
            stream_db.add(
                Message(
                    id=message_id,
                    session_id=session_id,
                    role=MessageRole.ASSISTANT.value,
                    content=content,
                    citations=[citation.model_dump(mode="json") for citation in citations],
                    tokens_used=(tokens_used if tokens_used is not None else count_tokens(content)),
                    latency_ms=latency_ms,
                )
            )
            stream_db.commit()
    except Exception:
        # A persistence hiccup must not corrupt the SSE stream the user is reading.
        logger.exception("Could not persist assistant message %s", message_id)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
