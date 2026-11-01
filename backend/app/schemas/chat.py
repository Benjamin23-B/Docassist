from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatSessionCreate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=50)


class ChatSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    doc_scope: list[uuid.UUID]
    created_at: datetime


class Citation(BaseModel):
    index: int
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    filename: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    score: float
    snippet: str


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    role: Literal["user", "assistant", "system"]
    content: str
    citations: list[Citation] = Field(default_factory=list)
    tokens_used: int | None = None
    latency_ms: int | None = None
    created_at: datetime


class ChatSessionDetail(ChatSessionRead):
    messages: list[ChatMessageRead] = Field(default_factory=list)


class ChatSessionListResponse(BaseModel):
    items: list[ChatSessionRead]
    total: int
    page: int
    page_size: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
