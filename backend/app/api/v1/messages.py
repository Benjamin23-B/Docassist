"""Message-level actions: answer feedback."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.db.models import ChatSession, Feedback, Message, MessageRole
from app.schemas.chat import FeedbackCreate, FeedbackRead

router = APIRouter()


@router.post(
    "/{message_id}/feedback",
    response_model=FeedbackRead,
    summary="Submit or update helpfulness feedback for an assistant message",
    responses={
        400: {"description": "Feedback can only target assistant messages"},
        404: {"description": "Message not found"},
    },
)
def submit_feedback(
    message_id: uuid.UUID,
    payload: FeedbackCreate,
    db: DbSession,
    current_user: CurrentUser,
) -> Feedback:
    message = db.scalar(
        select(Message)
        .join(ChatSession, Message.session_id == ChatSession.id)
        .where(
            Message.id == message_id,
            ChatSession.user_id == current_user.id,
        )
    )
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )
    if message.role != MessageRole.ASSISTANT.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback can only be submitted for assistant messages",
        )

    feedback = db.scalar(select(Feedback).where(Feedback.message_id == message.id))
    if feedback is None:
        feedback = Feedback(
            message_id=message.id,
            rating=payload.rating,
            comment=payload.comment,
        )
        db.add(feedback)
    else:
        # Feedback is a single verdict per answer; resubmitting replaces it so
        # users can correct a mistaken click or amend their comment.
        feedback.rating = payload.rating
        feedback.comment = payload.comment

    db.commit()
    db.refresh(feedback)
    return feedback
