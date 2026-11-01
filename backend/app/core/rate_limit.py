"""Per-identity fixed-window rate limiting.

Redis is the shared backend so limits hold across API replicas. If Redis is
unreachable the limiter falls back to a process-local window: traffic is still
bounded, and the API keeps serving instead of failing closed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from redis import Redis
from redis.exceptions import RedisError

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_INCR_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
return {count, ttl}
"""


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_after: int

    def headers(self) -> dict[str, str]:
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_after),
        }
        if not self.allowed:
            headers["Retry-After"] = str(self.reset_after)
        return headers


class RateLimitBackend(Protocol):
    def increment(self, key: str, window_seconds: int) -> tuple[int, int]: ...


class InMemoryBackend:
    """Process-local window used for tests and Redis outages."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def increment(self, key: str, window_seconds: int) -> tuple[int, int]:
        now = self._clock()
        with self._lock:
            count, reset_at = self._entries.get(key, (0, 0.0))
            if reset_at <= now:
                count = 0
                reset_at = now + window_seconds

            count += 1
            self._entries[key] = (count, reset_at)
            if len(self._entries) > 10_000:
                self._entries = {
                    entry_key: entry for entry_key, entry in self._entries.items() if entry[1] > now
                }
            return count, max(1, math.ceil(reset_at - now))


class RedisBackend:
    def __init__(self, redis_url: str) -> None:
        self._client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
        )

    def increment(self, key: str, window_seconds: int) -> tuple[int, int]:
        count, ttl = self._client.eval(_INCR_SCRIPT, 1, key, window_seconds)
        return int(count), max(1, int(ttl))


class ResilientBackend:
    """Try the shared backend first and degrade to the local window."""

    def __init__(self, primary: RateLimitBackend, fallback: RateLimitBackend) -> None:
        self._primary = primary
        self._fallback = fallback

    def increment(self, key: str, window_seconds: int) -> tuple[int, int]:
        try:
            return self._primary.increment(key, window_seconds)
        except (RedisError, OSError, TimeoutError):
            logger.warning("Rate-limit backend unavailable; using local fallback")
            return self._fallback.increment(key, window_seconds)


class RateLimiter:
    def __init__(
        self,
        *,
        limit: int,
        window_seconds: int,
        backend: RateLimitBackend,
        prefix: str,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._limit = limit
        self._window_seconds = window_seconds
        self._backend = backend
        self._prefix = prefix

    def check(self, identity: str) -> RateLimitResult:
        key = f"{self._prefix}:{identity}"
        count, reset_after = self._backend.increment(key, self._window_seconds)
        remaining = max(0, self._limit - count)
        return RateLimitResult(
            allowed=count <= self._limit,
            limit=self._limit,
            remaining=remaining,
            reset_after=reset_after,
        )


def _build_backend() -> RateLimitBackend:
    settings = get_settings()
    fallback = InMemoryBackend()
    if not settings.redis_url:
        return fallback
    try:
        return ResilientBackend(RedisBackend(settings.redis_url), fallback)
    except Exception:
        logger.warning("Could not initialise Redis rate limiter", exc_info=True)
        return fallback


@lru_cache(maxsize=1)
def get_global_rate_limiter() -> RateLimiter:
    settings = get_settings()
    return RateLimiter(
        limit=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
        backend=_build_backend(),
        prefix=f"{settings.rate_limit_key_prefix}:global",
    )


@lru_cache(maxsize=1)
def get_chat_rate_limiter() -> RateLimiter:
    settings = get_settings()
    return RateLimiter(
        limit=settings.chat_rate_limit_requests,
        window_seconds=settings.chat_rate_limit_window_seconds,
        backend=_build_backend(),
        prefix=f"{settings.rate_limit_key_prefix}:chat",
    )
