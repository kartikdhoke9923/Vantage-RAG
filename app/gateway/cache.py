"""Redis-backed prompt/response cache for the LLM gateway.

Cache key = sha256(feature | routing-model | prompt). Only populated when Redis
is reachable (reuses the cached startup probe) and GATEWAY_CACHE_ENABLED=true.
TTL is configurable via GATEWAY_CACHE_TTL (seconds).
"""
import hashlib
import json
from collections.abc import Callable

import logfire

from app.config import settings
from app.gateway import metrics as gateway_metrics
from app.services.health.connection_checker import redis_reachable


def _key(feature: str, routing_model: str, prompt: str) -> str:
    raw = json.dumps([feature, routing_model, prompt], sort_keys=True)
    return f"rag:llm:{hashlib.sha256(raw.encode()).hexdigest()}"


def _client():
    from redis import Redis

    return Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)


def cache_get(feature: str, routing_model: str, prompt: str) -> str | None:
    if not settings.GATEWAY_CACHE_ENABLED or not redis_reachable():
        return None
    try:
        value = _client().get(_key(feature, routing_model, prompt))
        return value.decode() if value else None
    except Exception as e:
        logfire.debug(f"Gateway cache read failed: {e}")
        return None


def cache_put(feature: str, routing_model: str, prompt: str, text: str) -> bool:
    if not settings.GATEWAY_CACHE_ENABLED or not redis_reachable():
        return False
    try:
        _client().set(_key(feature, routing_model, prompt), text, ex=settings.GATEWAY_CACHE_TTL)
        return True
    except Exception as e:
        logfire.debug(f"Gateway cache write failed: {e}")
        return False


def cached_completion(feature: str, routing_model: str, prompt: str, producer: Callable[[], str]) -> str:
    """Return a cached completion if present, else produce + cache it."""
    cached = cache_get(feature, routing_model, prompt)
    if cached is not None:
        gateway_metrics.GATEWAY_CACHE_HITS_TOTAL.labels(feature=feature).inc()
        logfire.info(f"⚡ Redis cache hit ({feature})")
        return cached

    result = producer()
    cache_put(feature, routing_model, prompt, result)
    return result


def cache_clear(feature: str | None = None) -> int:
    """Delete cached entries (optionally filtered by feature prefix)."""
    if not redis_reachable():
        return 0
    try:
        client = _client()
        pattern = "rag:llm:*"
        keys = list(client.scan_iter(match=pattern, count=500))
        if feature:
            prefix = f"rag:llm:{feature}"
            keys = [k for k in keys if k.decode().startswith(prefix)]
        if keys:
            client.delete(*keys)
        return len(keys)
    except Exception as e:
        logfire.warning(f"Gateway cache clear failed: {e}")
        return 0