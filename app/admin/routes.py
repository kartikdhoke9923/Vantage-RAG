"""Password/secret-gated admin endpoints for the site's control room.

Every route lives under /admin-api and requires the X-Admin-Token header to
match ADMIN_TOKEN. The site's Supabase login is the front door; this token is
the second layer that gates backend control.
"""
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app import db
from app.config import settings
from app.ingestion import scheduler

router = APIRouter(prefix="/admin-api", tags=["admin"])


def _require_admin(
    x_admin_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    if not settings.ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin API disabled (ADMIN_TOKEN unset).")
    token = x_admin_token
    if not token and authorization:
        token = authorization.removeprefix("Bearer ").strip()
    if token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token.")
    return True


class SettingsBody(BaseModel):
    cron_enabled: bool | None = None
    daily_hour: int | None = None


@router.get("/settings")
def get_settings(_: bool = Depends(_require_admin)):
    return {
        "cron_enabled": db.get_setting("cron_enabled", "true").lower() == "true",
        "daily_hour": settings.DAILY_INGEST_HOUR,
        "data_dir": settings.DATA_DIR,
        "logging_enabled": settings.ADMIN_LOGGING_ENABLED,
    }


@router.put("/settings")
def update_settings(body: SettingsBody, _: bool = Depends(_require_admin)):
    if body.cron_enabled is not None:
        db.set_setting("cron_enabled", "true" if body.cron_enabled else "false")
    return get_settings(_=_)


@router.post("/ingest/run")
def run_ingestion(_: bool = Depends(_require_admin)):
    job_id = scheduler.ingest_now("manual")
    return {"started": job_id is not None, "job_id": job_id}


@router.get("/ingest/jobs")
def ingest_jobs(limit: int = 20, _: bool = Depends(_require_admin)):
    return {"jobs": db.list_ingest_jobs(limit)}


@router.get("/chat/logs")
def chat_logs(limit: int = 50, _: bool = Depends(_require_admin)):
    return {"logs": db.list_chat_logs(limit)}


@router.get("/stats")
def stats(_: bool = Depends(_require_admin)):
    chroma_count = None
    chroma_error = None
    try:
        from app.services.health.connection_checker import check_all_connections

        connections = {
            name: {"healthy": r.healthy, "detail": r.detail}
            for name, r in check_all_connections().items()
        }
        from app.services.retrieval.chroma_client import get_or_create_collection

        chroma_count = get_or_create_collection().count()
    except Exception as e:  # noqa: BLE001 - stats must degrade gracefully
        chroma_error = str(e)
        connections = {}

    return {
        "chroma": {"count": chroma_count, "error": chroma_error},
        "chat": {
            "total": db.count_chat_logs(),
            "last_24h": db.count_chat_logs(24),
        },
        "last_job": (db.list_ingest_jobs(1) or [None])[0],
        "connections": connections,
    }


@router.get("/health")
def admin_health(_: bool = Depends(_require_admin)):
    return {"ok": True, "admin": "live"}