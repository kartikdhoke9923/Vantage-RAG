"""Daily ingestion scheduler for the backend process.

Purely threaded (no APScheduler dependency): a daemon loop computes the next
UTC tick based on DAILY_INGEST_HOUR and fires re-ingestion when enabled. Job
lifecycle is recorded in the ingest_jobs table for the admin page. Manual
"run now" triggers go through the same ingest_now() path.
"""
import threading
import time
from datetime import datetime, timedelta, timezone

import logfire

from app import db
from app.config import settings

_active: dict[int, str] = {}  # job_id -> kind, for in-flight jobs
_lock = threading.Lock()
_thread: threading.Thread | None = None


def ingest_now(kind: str = "manual") -> int | None:
    """Kick off an ingestion run in the background. Returns the job id, or the
    id of an already-running job, or None when a run is already in progress."""
    with _lock:
        if _active:
            return next(iter(_active), None)
        job_id = db.record_ingest_started(kind)
        if job_id is None:
            return None
        _active[job_id] = kind
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return job_id


def _run_job(job_id: int) -> None:
    try:
        from app.ingestion.processor import ingest_web_sources, run_universal_ingestion

        logfire.info("⏳ Ingestion job started.", job_id=job_id)
        run_universal_ingestion(settings.DATA_DIR, settings.DATA_SOURCE_TYPE, wipe=False)
        ingest_web_sources(wipe=False)
        db.record_ingest_finished(job_id, "ok", "Ingestion completed.")
        logfire.info("✅ Ingestion job completed.", job_id=job_id)
    except Exception as e:  # noqa: BLE001 - a failed tick must never take the API down
        logfire.error(f"❌ Ingestion job failed: {e}", job_id=job_id)
        db.record_ingest_finished(job_id, "error", str(e))
    finally:
        with _lock:
            _active.pop(job_id, None)


def _daily_loop() -> None:
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=settings.DAILY_INGEST_HOUR, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        time.sleep((target - now).total_seconds())

        enabled = db.get_setting("cron_enabled", "true").lower() == "true"
        if enabled:
            logfire.info("🕐 Daily ingestion tick — running.")
            ingest_now("daily")
        else:
            logfire.info("🕐 Daily ingestion tick — cron disabled, skipping.")


def start_scheduler() -> None:
    """Start the tick loop once per process (e.g. from FastAPI startup)."""
    global _thread
    if _thread is not None:
        return
    _thread = threading.Thread(target=_daily_loop, daemon=True)
    _thread.start()
    logfire.info(f"🕐 Daily ingestion scheduler running (UTC hour {settings.DAILY_INGEST_HOUR}).")


__all__ = ["ingest_now", "start_scheduler"]