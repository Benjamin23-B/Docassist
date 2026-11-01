"""Golden-set schema used by ``python -m evals.run_eval``."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    document: str | None = None
    keywords: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_relevance_signal(self) -> GoldenCase:
        self.document = self.document.strip() if self.document else None
        self.keywords = [keyword.strip() for keyword in self.keywords if keyword.strip()]
        self.expected_answer_contains = [
            expected.strip() for expected in self.expected_answer_contains if expected.strip()
        ]
        if not self.document and not self.keywords:
            raise ValueError(f"case {self.id!r} needs a document filename or at least one keyword")
        return self


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    cases: list[GoldenCase] = Field(min_length=1)


def load_dataset(path: Path) -> GoldenDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GoldenDataset.model_validate(payload)
