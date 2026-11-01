"""Markdown export for chat sessions.

Kept free of database and HTTP concerns so it can be unit tested directly.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from app.db.models import Message

_ROLE_LABELS = {
    "user": "You",
    "assistant": "Assistant",
    "system": "System",
}


def render_session_markdown(
    *,
    session_id: uuid.UUID,
    title: str,
    created_at: datetime,
    messages: Sequence[Message],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Session: `{session_id}`",
        f"- Created: {created_at.isoformat()}",
        "",
    ]

    if not messages:
        lines.extend(["_No messages in this conversation yet._", ""])

    for message in messages:
        lines.append(f"## {_ROLE_LABELS.get(message.role, message.role.title())}")
        lines.append("")
        lines.append(message.content.strip() or "_(empty message)_")
        lines.append("")

        citations = message.citations or []
        if citations:
            lines.append("**Sources**")
            lines.append("")
            for citation in citations:
                lines.append(_citation_line(citation))
            lines.append("")

        if message.feedback is not None:
            lines.append(_feedback_line(message.feedback))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_filename(title: str, *, session_id: uuid.UUID | None = None) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-._")
    if not slug:
        slug = "chat-session"
    slug = slug[:80]
    if session_id is not None:
        slug = f"{slug}-{str(session_id)[:8]}"
    return f"{slug}.md"


def _citation_line(citation: dict[str, Any]) -> str:
    index = citation.get("index", "?")
    filename = citation.get("filename", "document")
    location: list[str] = []
    if citation.get("page_number") is not None:
        location.append(f"page {citation['page_number']}")
    if citation.get("section"):
        location.append(f'"{citation["section"]}"')
    suffix = f" ({', '.join(location)})" if location else ""
    return f"- [{index}] {filename}{suffix}"


def _feedback_line(feedback: Any) -> str:
    verdict = "Helpful" if feedback.rating > 0 else "Not helpful"
    comment = f" — {feedback.comment}" if feedback.comment else ""
    return f"_Feedback: {verdict}{comment}_"
