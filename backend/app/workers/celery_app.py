from celery import Celery

from app.core.config import get_settings

settings = get_settings()
settings.validate_runtime()

celery_app = Celery(
    "rag",
    broker=settings.celery_broker_url or settings.redis_url,
    backend=settings.celery_result_backend or settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="ingestion",
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=False,
    broker_connection_retry_on_startup=True,
    result_expires=3600,
    task_time_limit=900,
    task_soft_time_limit=840,
)
