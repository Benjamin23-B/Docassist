"""Run a golden-set evaluation against a user's indexed documents.

Usage (from ``backend/``)::

    python -m evals.run_eval \
        --dataset evals/golden.example.json \
        --user-email you@example.com \
        --top-k 6

The runner uses the same retrieval, reranking and agent path as the chat API,
then scores the retrieved passages and an offline extractive answer. It exits
with status 1 when any configured quality threshold is missed, so it can run
in CI against a seeded tenant.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import User
from app.db.session import SessionLocal
from app.evaluation.dataset import GoldenCase, load_dataset
from app.evaluation.metrics import (
    answer_relevancy_score,
    faithfulness_score,
    mean,
    score_ranking,
)
from app.rag.agent import run_agent
from app.rag.llm import DeterministicLLMProvider, SourcePassage
from app.rag.reranker import rerank_chunks
from app.rag.retriever import RetrievedChunk, retrieve_chunks


@dataclass(frozen=True, slots=True)
class CaseResult:
    id: str
    question: str
    hit_at_k: float
    reciprocal_rank: float
    precision_at_k: float
    faithfulness: float
    answer_relevancy: float
    answer_contains: bool | None
    retrieved: int
    relevant: int
    retrieval_ms: int
    total_ms: int
    answer_preview: str


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset = load_dataset(Path(args.dataset))

    with SessionLocal() as db:
        user = _find_user(db, email=args.user_email, user_id=args.user_id)
        if user is None:
            print("No matching user found. Pass --user-email or --user-id.", file=sys.stderr)
            return 2

        results = [
            _evaluate_case(
                db,
                user_id=user.id,
                case=case,
                top_k=args.top_k,
                use_agent=not args.no_agent,
            )
            for case in dataset.cases
        ]

    summary = _summarize(results, top_k=args.top_k)
    report = {
        "dataset": dataset.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "top_k": args.top_k,
        "agent_enabled": not args.no_agent,
        "summary": summary,
        "cases": [asdict(result) for result in results],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    failures = _threshold_failures(summary, args)
    if failures:
        for name, minimum, actual in failures:
            print(f"threshold missed: {name}={actual:.3f} < {minimum:.3f}", file=sys.stderr)
        return 1
    return 0


def _evaluate_case(
    db: Session,
    *,
    user_id: uuid.UUID,
    case: GoldenCase,
    top_k: int,
    use_agent: bool,
) -> CaseResult:
    settings = get_settings()
    reranking = settings.reranker_provider != "none"
    candidate_k = settings.rerank_candidate_k if reranking else settings.retrieval_top_k

    started = time.perf_counter()
    if use_agent and settings.agent_enabled:
        agent_result = run_agent(
            db,
            user_id=user_id,
            question=case.question,
            document_ids=[],
            candidate_k=candidate_k,
            top_k=top_k,
        )
        chunks = agent_result.chunks
    else:
        chunks = retrieve_chunks(
            db,
            user_id=user_id,
            query=case.question,
            top_k=candidate_k,
        )
        chunks = rerank_chunks(chunks, query=case.question, top_k=top_k)
    retrieval_ms = int((time.perf_counter() - started) * 1000)

    relevances = [_is_relevant(case, chunk) for chunk in chunks]
    ranking = score_ranking(relevances)

    passages = [
        SourcePassage(
            index=index,
            source=f"{chunk.filename} · chunk {chunk.chunk_index}",
            text=chunk.content,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    answer = DeterministicLLMProvider().compose_answer(
        question=case.question,
        passages=passages,
    )
    answer_contains = None
    if case.expected_answer_contains:
        lowered_answer = answer.casefold()
        answer_contains = all(
            expected.casefold() in lowered_answer for expected in case.expected_answer_contains
        )

    return CaseResult(
        id=case.id,
        question=case.question,
        hit_at_k=ranking.hit_at_k,
        reciprocal_rank=ranking.reciprocal_rank,
        precision_at_k=ranking.precision_at_k,
        faithfulness=faithfulness_score(answer, [chunk.content for chunk in chunks]),
        answer_relevancy=answer_relevancy_score(case.question, answer),
        answer_contains=answer_contains,
        retrieved=len(chunks),
        relevant=sum(relevances),
        retrieval_ms=retrieval_ms,
        total_ms=int((time.perf_counter() - started) * 1000),
        answer_preview=" ".join(answer.split())[:280],
    )


def _is_relevant(case: GoldenCase, chunk: RetrievedChunk) -> bool:
    if case.document:
        expected = case.document.casefold()
        filename_matches = Path(chunk.filename).name.casefold() == expected
        path_matches = chunk.filename.casefold() == expected
        if not (filename_matches or path_matches):
            return False
    if case.keywords:
        content = chunk.content.casefold()
        if not any(keyword.casefold() in content for keyword in case.keywords):
            return False
    return True


def _summarize(results: Sequence[CaseResult], *, top_k: int) -> dict[str, Any]:
    checked = [result for result in results if result.answer_contains is not None]
    return {
        "cases": len(results),
        "top_k": top_k,
        "hit_rate_at_k": round(mean(result.hit_at_k for result in results), 4),
        "mrr": round(mean(result.reciprocal_rank for result in results), 4),
        "precision_at_k": round(mean(result.precision_at_k for result in results), 4),
        "faithfulness": round(mean(result.faithfulness for result in results), 4),
        "answer_relevancy": round(mean(result.answer_relevancy for result in results), 4),
        "answer_contains_rate": (
            round(mean(1.0 if result.answer_contains else 0.0 for result in checked), 4)
            if checked
            else None
        ),
        "mean_retrieval_ms": round(mean(result.retrieval_ms for result in results), 1),
        "mean_total_ms": round(mean(result.total_ms for result in results), 1),
    }


def _threshold_failures(
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> list[tuple[str, float, float]]:
    checks = [
        ("hit_rate_at_k", args.min_hit_rate),
        ("mrr", args.min_mrr),
        ("faithfulness", args.min_faithfulness),
        ("answer_relevancy", args.min_answer_relevancy),
    ]
    return [
        (name, minimum, float(summary[name]))
        for name, minimum in checks
        if minimum is not None and float(summary[name]) < minimum
    ]


def _find_user(
    db: Session,
    *,
    email: str | None,
    user_id: str | None,
) -> User | None:
    if email:
        return db.scalar(select(User).where(func.lower(User.email) == email.strip().lower()))
    if user_id:
        try:
            parsed = uuid.UUID(user_id)
        except ValueError:
            return None
        return db.get(User, parsed)
    return None


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="evals/golden.example.json",
        help="Path to a golden-set JSON file.",
    )
    parser.add_argument("--user-email", help="Tenant whose documents should be evaluated.")
    parser.add_argument("--user-id", help="Tenant UUID (alternative to --user-email).")
    parser.add_argument("--top-k", type=int, default=6, help="Passages scored per question.")
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Bypass the configured agent planner and evaluate retrieval only.",
    )
    parser.add_argument("--output", help="Optional path for the full JSON report.")
    parser.add_argument("--min-hit-rate", type=float)
    parser.add_argument("--min-mrr", type=float)
    parser.add_argument("--min-faithfulness", type=float)
    parser.add_argument("--min-answer-relevancy", type=float)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
