"""Candidate reranking.

The deterministic reranker is a lexical cross-check that works offline. The
Cohere provider calls a hosted rerank model for production quality; failures
fall back to the fused retrieval order so chat never breaks on a reranker
outage.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from typing import Protocol

from app.core.config import get_settings
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "but",
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


class RerankerError(RuntimeError):
    """The reranker is misconfigured or the provider request failed."""


class Reranker(Protocol):
    name: str

    def rerank(
        self,
        chunks: Sequence[RetrievedChunk],
        query: str,
        *,
        top_k: int,
    ) -> list[RetrievedChunk]: ...


class DeterministicReranker:
    """Score candidates by lexical coverage, metadata overlap and prior rank."""

    name = "deterministic"

    def rerank(
        self,
        chunks: Sequence[RetrievedChunk],
        query: str,
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        query_terms = _terms(query)
        normalised_query = " ".join(query.lower().split())

        rescored = [
            replace(
                chunk,
                rerank_score=self._score(
                    chunk,
                    query_terms=query_terms,
                    normalised_query=normalised_query,
                ),
            )
            for chunk in chunks
        ]
        rescored.sort(
            key=lambda chunk: (chunk.rerank_score or 0.0, chunk.score),
            reverse=True,
        )
        return rescored[:top_k]

    def _score(
        self,
        chunk: RetrievedChunk,
        *,
        query_terms: set[str],
        normalised_query: str,
    ) -> float:
        prior = min(max(chunk.score, 0.0), 1.0)
        if not query_terms:
            return prior

        content_terms = _terms(chunk.content)
        coverage = len(query_terms & content_terms) / len(query_terms)

        metadata_terms = _terms(f"{chunk.filename} {chunk.section or ''}")
        metadata_coverage = len(query_terms & metadata_terms) / len(query_terms)

        phrase_bonus = 0.0
        if len(normalised_query) >= 8:
            phrase_bonus = (
                1.0 if normalised_query in " ".join(chunk.content.lower().split()) else 0.0
            )

        score = 0.6 * coverage + 0.15 * metadata_coverage + 0.1 * phrase_bonus + 0.15 * prior
        return round(min(max(score, 0.0), 1.0), 4)


class CohereReranker:
    """Hosted cross-encoder reranking through the Cohere rerank API."""

    name = "cohere"

    def __init__(self, *, api_key: str, model: str, base_url: str) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = f"{base_url.rstrip('/')}/rerank"

    def rerank(
        self,
        chunks: Sequence[RetrievedChunk],
        query: str,
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        payload = {
            "model": self._model,
            "query": query,
            "documents": [chunk.content for chunk in chunks],
            "top_n": min(top_k, len(chunks)),
        }
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = json.load(response)
        except Exception as exc:
            raise RerankerError(f"Cohere rerank request failed: {exc}") from exc

        results = body.get("results")
        if not isinstance(results, list):
            raise RerankerError("Cohere rerank response did not contain results")

        reranked: list[RetrievedChunk] = []
        for result in results:
            try:
                index = int(result["index"])
                relevance = float(result["relevance_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < len(chunks):
                reranked.append(replace(chunks[index], rerank_score=min(max(relevance, 0.0), 1.0)))
        return reranked[:top_k]


def rerank_chunks(
    chunks: Sequence[RetrievedChunk],
    *,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Rerank candidates, degrading to the retrieval order on any failure."""
    settings = get_settings()
    if not chunks or settings.reranker_provider == "none":
        return list(chunks)[:top_k]

    try:
        return get_reranker().rerank(chunks, query, top_k=top_k)
    except Exception:
        logger.warning("Reranking failed; keeping retrieval order", exc_info=True)
        return list(chunks)[:top_k]


def _terms(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


@lru_cache(maxsize=1)
def get_reranker() -> DeterministicReranker | CohereReranker:
    settings = get_settings()
    if settings.reranker_provider == "cohere":
        if not settings.cohere_api_key:
            raise RerankerError("COHERE_API_KEY is required when RERANKER_PROVIDER=cohere")
        return CohereReranker(
            api_key=settings.cohere_api_key,
            model=settings.cohere_rerank_model,
            base_url=settings.cohere_base_url,
        )
    return DeterministicReranker()
