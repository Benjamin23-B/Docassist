import json

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import GoldenDataset, load_dataset
from app.evaluation.metrics import (
    answer_relevancy_score,
    faithfulness_score,
    score_ranking,
)


def test_score_ranking_finds_first_relevant_position() -> None:
    score = score_ranking([False, False, True, True])

    assert score.hit_at_k == 1.0
    assert score.reciprocal_rank == pytest.approx(1 / 3)
    assert score.precision_at_k == pytest.approx(0.5)


def test_score_ranking_handles_no_matches() -> None:
    score = score_ranking([False, False])

    assert score.hit_at_k == 0.0
    assert score.reciprocal_rank == 0.0
    assert score.precision_at_k == 0.0


def test_score_ranking_handles_empty_results() -> None:
    score = score_ranking([])

    assert score.hit_at_k == 0.0
    assert score.reciprocal_rank == 0.0
    assert score.precision_at_k == 0.0


def test_faithfulness_rewards_supported_answers() -> None:
    contexts = ["Employees receive twenty-five days of annual leave each year."]

    supported = faithfulness_score("Employees receive annual leave days [1].", contexts)
    invented = faithfulness_score("Employees receive cryptocurrency bonuses quarterly.", contexts)

    assert supported == 1.0
    assert invented < supported


def test_answer_relevancy_tracks_question_terms() -> None:
    relevant = answer_relevancy_score(
        "How many annual leave days?",
        "Employees receive twenty-five annual leave days.",
    )
    off_topic = answer_relevancy_score(
        "How many annual leave days?",
        "The security badge policy was updated.",
    )

    assert relevant > off_topic


def _dataset_payload() -> dict:
    return {
        "name": "smoke",
        "cases": [
            {
                "id": "leave",
                "question": "How many leave days?",
                "document": "handbook.md",
                "keywords": ["annual leave"],
            }
        ],
    }


def test_load_dataset_round_trips(tmp_path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(_dataset_payload()), encoding="utf-8")

    dataset = load_dataset(path)

    assert isinstance(dataset, GoldenDataset)
    assert dataset.cases[0].id == "leave"


def test_dataset_rejects_cases_without_relevance_signal() -> None:
    payload = _dataset_payload()
    payload["cases"][0].pop("document")
    payload["cases"][0].pop("keywords")

    with pytest.raises(ValidationError, match="needs a document filename"):
        GoldenDataset.model_validate(payload)
