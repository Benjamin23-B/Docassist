"""Offline evaluation helpers for retrieval and grounded answers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+")
_CITATION_RE = re.compile(r"\[\d+\]")

_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "could",
    "did",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "its",
    "may",
    "much",
    "many",
    "not",
    "our",
    "should",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


@dataclass(frozen=True, slots=True)
class RankingScore:
    hit_at_k: float
    reciprocal_rank: float
    precision_at_k: float


def score_ranking(relevances: Sequence[bool]) -> RankingScore:
    """Score an ordered relevance list for one query."""
    if not relevances:
        return RankingScore(hit_at_k=0.0, reciprocal_rank=0.0, precision_at_k=0.0)

    first_rank = next(
        (index for index, relevant in enumerate(relevances, start=1) if relevant),
        None,
    )
    return RankingScore(
        hit_at_k=1.0 if first_rank is not None else 0.0,
        reciprocal_rank=(1.0 / first_rank) if first_rank is not None else 0.0,
        precision_at_k=sum(relevances) / len(relevances),
    )


def faithfulness_score(answer: str, contexts: Iterable[str]) -> float:
    """Approximate groundedness: share of answer terms found in the context.

    This is a deterministic offline proxy, not an LLM judge. It catches answers
    that introduce many unsupported terms while remaining cheap enough for CI.
    """
    answer_terms = _terms(_CITATION_RE.sub(" ", answer))
    if not answer_terms:
        return 0.0
    context_terms = _terms(" ".join(contexts))
    return len(answer_terms & context_terms) / len(answer_terms)


def answer_relevancy_score(question: str, answer: str) -> float:
    """Approximate relevancy: share of question terms echoed in the answer."""
    question_terms = _terms(question)
    if not question_terms:
        return 0.0
    answer_terms = _terms(_CITATION_RE.sub(" ", answer))
    return len(question_terms & answer_terms) / len(question_terms)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _terms(text: str) -> set[str]:
    return {
        word for word in _WORD_RE.findall(text.lower()) if len(word) > 2 and word not in _STOPWORDS
    }
