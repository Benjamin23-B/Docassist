"""Dead-letter queue for documents that exhaust their ingestion retries."""

from __future__ import annotations

import json
import logging
from typing import Any

import redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

MAX_DLQ_ENTRIES = 1000


def push_dead_letter(payload: dict[str, Any]) -> None:
    settings = get_settings()
    try:
        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        client.rpush(settings.dlq_key, json.dumps(payload, default=str))
        client.ltrim(settings.dlq_key, -MAX_DLQ_ENTRIES, -1)
    except Exception as exc:
        # Losing a DLQ record must never mask the original ingestion failure.
        logger.error("Could not push dead-letter record: %s", exc)
