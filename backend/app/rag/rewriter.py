"""Contextual query rewriting for follow-up questions.

Follow-ups such as "how many days is it?" retrieve poorly on their own because
pronouns and omitted subjects carry no lexical signal. Rewriters turn them into
standalone search queries using the recent conversation; the deterministic
provider keeps this usable offline.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from app.core.config import get_settings
from app.rag.llm import ChatTurn

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")

_FOLLOW_UP_STARTS = {
    "and",
    "also",
    "another",
    "but",
    "more",
    "same",
    "so",
    "then",
}

_FOLLOW_UP_PHRASES = (
    "what about",
    "how about",
    "what if",
    "and what",
    "and how",
    "same for",
)

_PRONOUNS = {
    "above",
    "another",
    "former",
    "her",
    "him",
    "his",
    "it",
    "its",
    "latter",
    "one",
    "ones",
    "she",
    "he",
    "that",
    "their",
    "them",
    "these",
    "they",
    "this",
    "those",
}

_PRONOMINAL_REFERENCES = _PRONOUNS | {"there", "same"}

MAX_CONTEXT_CHARS = 320


class QueryRewriteError(RuntimeError):
    """The configured query rewriter could not produce a query."""


def is_follow_up(question: str, previous_question: str | None) -> bool:
    """Heuristically decide whether ``question`` depends on prior context."""
    if not previous_question:
        return False

    normalised = " ".join(question.split()).lower()
    if not normalised:
        return False

    words = _WORD_RE.findall(normalised)
    if not words:
        return False

    if len(words) <= 3:
        return True
    if words[0] in _FOLLOW_UP_STARTS:
        return True
    if any(normalised.startswith(phrase) for phrase in _FOLLOW_UP_PHRASES):
        return True
    return bool(_PRONOMINAL_REFERENCES & set(words))


class QueryRewriter(Protocol):
    name: str

    def rewrite(self, question: str, history: Sequence[ChatTurn]) -> str: ...


class DeterministicQueryRewriter:
    """Append the previous question when the current one is a follow-up."""

    name = "deterministic"

    def rewrite(self, question: str, history: Sequence[ChatTurn]) -> str:
        previous = _previous_questions(history)
        if not previous or not is_follow_up(question, previous[-1]):
            return question

        context = " ".join(previous[-1].split())[:MAX_CONTEXT_CHARS]
        current = " ".join(question.split())
        if not context or context.lower() in current.lower():
            return current
        return f"{context} {current}"


class OpenAIQueryRewriter:
    """Ask a chat model to resolve references into a standalone query."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        history_limit: int = 4,
    ) -> None:
        from openai import OpenAI

        self._model = model
        self._history_limit = max(history_limit, 0)
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            max_retries=1,
            timeout=30.0,
        )

    def rewrite(self, question: str, history: Sequence[ChatTurn]) -> str:
        transcript = _render_transcript(history, limit=self._history_limit)
        if not transcript:
            return question

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                max_tokens=160,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user's latest message as a standalone "
                            "search query. Use the conversation only to resolve "
                            "references such as 'it', 'they', 'that' or omitted "
                            "subjects. Preserve names, numbers and constraints. "
                            "Do not answer the question. Return only the query."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Conversation:\n{transcript}\n\nLatest message: {question}",
                    },
                ],
            )
        except Exception as exc:
            raise QueryRewriteError(f"Query rewrite request failed: {exc}") from exc

        if not response.choices:
            return question
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or question


def rewrite_query(question: str, history: Sequence[ChatTurn]) -> str:
    """Rewrite a follow-up, falling back to the original on any failure."""
    settings = get_settings()
    if not settings.query_rewrite_enabled or not history:
        return question

    original = " ".join(question.split())
    try:
        rewritten = get_query_rewriter().rewrite(original, history)
    except Exception:
        logger.warning("Query rewriting failed; using the original question", exc_info=True)
        return original

    cleaned = " ".join(rewritten.split())
    return cleaned or original


def _previous_questions(history: Sequence[ChatTurn]) -> list[str]:
    return [
        turn.content.strip() for turn in history if turn.role == "user" and turn.content.strip()
    ]


def _render_transcript(history: Sequence[ChatTurn], *, limit: int) -> str:
    if limit <= 0:
        return ""
    lines: list[str] = []
    for turn in list(history)[-limit:]:
        if turn.role not in {"user", "assistant"} or not turn.content.strip():
            continue
        speaker = "User" if turn.role == "user" else "Assistant"
        text = " ".join(turn.content.split())[:600]
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_query_rewriter() -> DeterministicQueryRewriter | OpenAIQueryRewriter:
    settings = get_settings()
    if settings.query_rewrite_provider == "openai":
        if not settings.openai_api_key:
            raise QueryRewriteError("OPENAI_API_KEY is required when QUERY_REWRITE_PROVIDER=openai")
        return OpenAIQueryRewriter(
            api_key=settings.openai_api_key,
            model=settings.query_rewrite_model or settings.llm_model,
            base_url=settings.openai_base_url,
            history_limit=settings.query_rewrite_history_limit,
        )
    return DeterministicQueryRewriter()
