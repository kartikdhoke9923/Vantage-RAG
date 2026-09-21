"""Health check endpoint for the Enterprise RAG API."""
import logfire
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.health.connection_checker import (
    check_all_connections,
    log_connection_summary,
)

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Report liveness plus the reachability of external services."""
    with logfire.span("🩺 Health Check"):
        results = check_all_connections()
        summary = {name: r.healthy for name, r in results.items()}
        all_healthy = all(summary.values())
        return JSONResponse(
            status_code=200 if all_healthy else 503,
            content={"status": "ok" if all_healthy else "degraded", "services": summary},
        )


@router.get("/health/summary")
def health_summary():
    """Human-readable summary of external service connectivity."""
    results = check_all_connections()
    lines = log_connection_summary(results)
    return {"lines": lines}