"""External-service connectivity checks used at startup and by /health."""
import time
from dataclasses import dataclass

import logfire


@dataclass
class ConnectionResult:
    name: str
    healthy: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.healthy


# Redis is probed by both the health module and the rate limiter at startup.
# Cache the result so it is only tested once per process/interval instead of
# blocking boot with a second connect timeout.
_REDIS_PROBE_TTL_SECONDS = 30
_REDIS_PROBE_CACHE: dict = {"at": 0.0, "result": None}


def _check_chroma() -> ConnectionResult:
    """Verify the Chroma Cloud collection is reachable and exists."""
    try:
        from app.services.retrieval.chroma_client import get_or_create_collection

        collection = get_or_create_collection()
        count = collection.count()
        return ConnectionResult(name="chroma", healthy=True, detail=f"collection ready, {count} chunks")
    except Exception as e:
        logfire.error(f"❌ Chroma Cloud unreachable: {e}")
        return ConnectionResult(name="chroma", healthy=False, detail=str(e))


def _check_redis() -> ConnectionResult:
    """Best-effort Redis connectivity (used for rate limiting), cached per boot."""
    now = time.monotonic()
    if _REDIS_PROBE_CACHE["result"] is not None and now - _REDIS_PROBE_CACHE["at"] < _REDIS_PROBE_TTL_SECONDS:
        return _REDIS_PROBE_CACHE["result"]
    try:
        import redis

        from app.config import settings

        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        client.ping()
        result = ConnectionResult(name="redis", healthy=True, detail="pong")
    except Exception as e:
        result = ConnectionResult(name="redis", healthy=False, detail=str(e))
    _REDIS_PROBE_CACHE.update(at=now, result=result)
    return result


def redis_reachable() -> bool:
    """Cached Redis probe for consumers (e.g. the rate limiter)."""
    return _check_redis().healthy


def check_all_connections() -> dict[str, ConnectionResult]:
    """Run connectivity probes for every external service the pipeline needs."""
    results: dict[str, ConnectionResult] = {
        "chroma": _check_chroma(),
        "redis": _check_redis(),
    }
    return results


def log_connection_summary(results: dict[str, ConnectionResult]) -> list[str]:
    """Log a one-line summary per service; returns the human-readable lines."""
    lines = []
    for name, result in results.items():
        status = "✅" if result.healthy else "❌"
        line = f"{status} {name}: {result.detail}"
        lines.append(line)
        if result.healthy:
            logfire.info(line)
        else:
            logfire.warning(line)
    return lines