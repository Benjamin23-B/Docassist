import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.db.models import Feedback, Message
from app.schemas.chat import (
    AskRequest,
    ChatMessageRead,
    ChatSessionCreate,
    ChatSessionUpdate,
    FeedbackCreate,
)
from app.services.export import export_filename, render_session_markdown


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


def test_session_update_strips_and_rejects_blank_titles() -> None:
    assert ChatSessionUpdate(title="  Benefits  ").title == "Benefits"
    with pytest.raises(ValidationError):
        ChatSessionUpdate(title="   ")


@pytest.mark.parametrize("rating", [-1, 1])
def test_feedback_create_accepts_binary_ratings(rating: int) -> None:
    payload = FeedbackCreate(rating=rating, comment="  useful context  ")

    assert payload.rating == rating
    assert payload.comment == "useful context"


@pytest.mark.parametrize("rating", [0, 2, -2])
def test_feedback_create_rejects_non_binary_ratings(rating: int) -> None:
    with pytest.raises(ValidationError):
        FeedbackCreate(rating=rating)


def test_feedback_create_normalises_blank_comments() -> None:
    assert FeedbackCreate(rating=1, comment="   ").comment is None
    assert FeedbackCreate(rating=1).comment is None


def test_message_read_includes_existing_feedback() -> None:
    feedback_id = uuid.uuid4()
    message = ChatMessageRead.model_validate(
        {
            "id": uuid.uuid4(),
            "session_id": uuid.uuid4(),
            "role": "assistant",
            "content": "Answer",
            "citations": [],
            "feedback": {
                "id": feedback_id,
                "message_id": uuid.uuid4(),
                "rating": -1,
                "comment": "Missed the deadline",
                "created_at": datetime.now(UTC),
            },
            "created_at": datetime.now(UTC),
        }
    )

    assert message.feedback is not None
    assert message.feedback.id == feedback_id
    assert message.feedback.rating == -1


def test_render_session_markdown_includes_messages_citations_and_feedback() -> None:
    session_id = uuid.uuid4()
    message = Message(
        session_id=session_id,
        role="assistant",
        content="Twenty-five days [1].",
        citations=[
            {
                "index": 1,
                "filename": "handbook.md",
                "page_number": 3,
                "section": "Leave",
            }
        ],
    )
    message.feedback = Feedback(rating=1, comment="Clear")

    markdown = render_session_markdown(
        session_id=session_id,
        title="Leave policy",
        created_at=datetime.now(UTC),
        messages=[message],
    )

    assert markdown.startswith("# Leave policy")
    assert "## Assistant" in markdown
    assert "Twenty-five days [1]." in markdown
    assert '[1] handbook.md (page 3, "Leave")' in markdown
    assert "_Feedback: Helpful — Clear_" in markdown


def test_render_session_markdown_handles_empty_sessions() -> None:
    markdown = render_session_markdown(
        session_id=uuid.uuid4(),
        title="New chat",
        created_at=datetime.now(UTC),
        messages=[],
    )

    assert "_No messages in this conversation yet._" in markdown


def test_export_filename_sanitises_session_titles() -> None:
    assert export_filename("Q3 / Benefits?", session_id=uuid.uuid4()).endswith(".md")
    assert "/" not in export_filename("Q3 / Benefits?")
    assert export_filename("   ") == "chat-session.md"
