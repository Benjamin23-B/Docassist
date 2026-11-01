from app.workers import tasks  # noqa: F401 - registers the task
from app.workers.celery_app import celery_app
from app.workers.pipeline import (
    PermanentIngestionError,
    TransientIngestionError,
)


def test_ingestion_task_is_registered() -> None:
    assert "ingestion.ingest_document" in celery_app.tasks


def test_worker_uses_dedicated_queue() -> None:
    assert celery_app.conf.task_default_queue == "ingestion"


def test_pipeline_errors_are_distinct() -> None:
    assert not issubclass(PermanentIngestionError, TransientIngestionError)
    assert issubclass(PermanentIngestionError, RuntimeError)
    assert issubclass(TransientIngestionError, RuntimeError)
