from types import SimpleNamespace

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

import app.main as main
from app.core.rate_limit import (
    InMemoryBackend,
    RateLimiter,
    ResilientBackend,
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_fixed_window_allows_limit_then_rejects() -> None:
    clock = _FakeClock()
    limiter = RateLimiter(
        limit=2,
        window_seconds=60,
        backend=InMemoryBackend(clock=clock),
        prefix="test",
    )

    first = limiter.check("user:1")
    second = limiter.check("user:1")
    third = limiter.check("user:1")

    assert first.allowed is True
    assert first.remaining == 1
    assert second.allowed is True
    assert second.remaining == 0
    assert third.allowed is False
    assert third.remaining == 0
    assert third.headers()["Retry-After"] == "60"


def test_window_resets_after_expiry() -> None:
    clock = _FakeClock()
    limiter = RateLimiter(
        limit=1,
        window_seconds=10,
        backend=InMemoryBackend(clock=clock),
        prefix="test",
    )

    assert limiter.check("user:1").allowed is True
    assert limiter.check("user:1").allowed is False

    clock.now += 11

    assert limiter.check("user:1").allowed is True


def test_limiter_keeps_identities_separate() -> None:
    limiter = RateLimiter(
        limit=1,
        window_seconds=60,
        backend=InMemoryBackend(),
        prefix="test",
    )

    assert limiter.check("user:1").allowed is True
    assert limiter.check("user:2").allowed is True
    assert limiter.check("user:1").allowed is False


def test_resilient_backend_falls_back_when_redis_fails() -> None:
    class BrokenBackend:
        def increment(self, key: str, window_seconds: int) -> tuple[int, int]:
            raise RedisConnectionError("redis down")

    fallback = InMemoryBackend()
    backend = ResilientBackend(BrokenBackend(), fallback)

    assert backend.increment("key", 60) == (1, 60)
    assert fallback.increment("key", 60) == (2, 60)


def test_middleware_rejects_requests_over_the_limit(monkeypatch) -> None:
    limiter = RateLimiter(
        limit=1,
        window_seconds=60,
        backend=InMemoryBackend(),
        prefix="test:global",
    )
    monkeypatch.setattr(
        main, "settings", SimpleNamespace(rate_limit_enabled=True, api_v1_prefix="/api/v1")
    )
    monkeypatch.setattr(main, "get_global_rate_limiter", lambda: limiter)

    client = TestClient(main.app)
    first = client.get("/api/v1/openapi.json")
    second = client.get("/api/v1/openapi.json")

    assert first.status_code == 200
    assert first.headers["x-ratelimit-limit"] == "1"
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"


def test_result_headers_include_limit_state() -> None:
    limiter = RateLimiter(
        limit=5,
        window_seconds=30,
        backend=InMemoryBackend(),
        prefix="test",
    )

    result = limiter.check("ip:127.0.0.1")

    assert result.headers() == {
        "X-RateLimit-Limit": "5",
        "X-RateLimit-Remaining": "4",
        "X-RateLimit-Reset": "30",
    }
