import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas.chat import (
    AskRequest,
    ChatMessageRead,
    ChatSessionCreate,
)


def test_chat_session_create_defaults() -> None:
    payload = ChatSessionCreate()

    assert payload.title is None
    assert payload.document_ids == []


def test_chat_session_create_accepts_document_scope() -> None:
    document_id = uuid.uuid4()
    payload = ChatSessionCreate(title="Benefits", document_ids=[document_id])

    assert payload.title == "Benefits"
    assert payload.document_ids == [document_id]


def test_ask_request_rejects_empty_and_overlong_questions() -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="")
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 4001)


def test_message_read_parses_stored_citation_json() -> None:
    session_id = uuid.uuid4()
    message = ChatMessageRead.model_validate(
        {
            "id": uuid.uuid4(),
            "session_id": session_id,
            "role": "assistant",
            "content": "Twenty-five days [1].",
            "citations": [
                {
                    "index": 1,
                    "document_id": str(uuid.uuid4()),
                    "chunk_id": str(uuid.uuid4()),
                    "filename": "handbook.md",
                    "chunk_index": 0,
                    "page_number": 1,
                    "section": "Leave",
                    "score": 0.82,
                    "snippet": "Twenty-five days of leave.",
                }
            ],
            "tokens_used": 12,
            "latency_ms": 250,
            "created_at": datetime.now(UTC),
        }
    )

    assert message.role == "assistant"
    assert message.citations[0].filename == "handbook.md"
    assert message.session_id == session_id
