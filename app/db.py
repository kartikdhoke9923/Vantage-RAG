"""Neon Postgres access for the admin layer.

Stores chat logs, ingestion job history and app settings in tables created
idempotently at startup. Every function degrades to a safe no-op when
POSTGRES_URI is missing (local dev without a DB), so the RAG pipeline is
never blocked by logging failures.
"""
import threading
import time
from typing import Any

import logfire

from app.config import settings

_pool = None
_pool_lock = threading.Lock()
# Back-off when pool creation fails (e.g. Neon briefly unreachable at boot).
_negative_until = 0.0


def _get_pool():
    """Lazily build the psycopg connection pool (once per process)."""
    global _pool, _negative_until
    if not settings.POSTGRES_URI or time.monotonic() < _negative_until:
        return None
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                try:
                    from psycopg_pool import ConnectionPool

                    _pool = ConnectionPool(
                        settings.POSTGRES_URI,
                        min_size=1,
                        max_size=4,
                        open=False,  # lazily opens on first checkout
                        timeout=10,
                        kwargs={"connect_timeout": 5, "options": "-c statement_timeout=3000"},
                    )
                except Exception as e:
                    _negative_until = time.monotonic() + 30
                    logfire.warning(f"⚠️ DB pool init failed (retry in 30s): {e}")
                    return None
    return _pool


def _conn():
    """Yield a connection, or None when the DB is unavailable."""
    pool = _get_pool()
    if pool is None:
        return None
    try:
        return pool.connection()
    except Exception as e:
        logfire.warning(f"⚠️ DB pool checkout failed: {e}")
        return None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_logs (
    id           BIGSERIAL PRIMARY KEY,
    thread_id    TEXT,
    question     TEXT,
    answer       TEXT,
    status       TEXT,
    latency_ms   INTEGER,
    cache_hit    TEXT,
    source_count INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id          BIGSERIAL PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    message     TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def init_db() -> bool:
    """Create tables if they don't exist. Returns False when no DB is set."""
    conn = _conn()
    if conn is None:
        logfire.warning("🪙 POSTGRES_URI not set — admin logging tables not created.")
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)
        conn.commit()
        logfire.info("🪙 Admin tables ready (chat_logs, ingest_jobs, app_settings).")
        return True
    except Exception as e:
        logfire.error(f"⚠️ Could not initialise admin tables: {e}")
        return False
    finally:
        conn.close()


# ── chat_logs ─────────────────────────────────────────────────────────────────


def log_chat(
    thread_id: str,
    question: str,
    answer: str,
    status: str,
    latency_ms: int,
    cache_hit: str = "none",
    source_count: int = 0,
) -> None:
    """Append one chat exchange to chat_logs (best-effort, fire-and-forget)."""
    if not settings.ADMIN_LOGGING_ENABLED:
        return
    conn = _conn()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO chat_logs
                   (thread_id, question, answer, status, latency_ms, cache_hit, source_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    (thread_id or "")[:120],
                    (question or "")[:2000],
                    (answer or "")[:20000],
                    (status or "")[:300],
                    int(latency_ms),
                    cache_hit[:40],
                    int(source_count or 0),
                ),
            )
        conn.commit()
    except Exception as e:
        logfire.warning(f"⚠️ chat_log insert failed: {e}")
    finally:
        conn.close()


def list_chat_logs(limit: int = 50) -> list[dict[str, Any]]:
    conn = _conn()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, thread_id, question, status, latency_ms, cache_hit, source_count, created_at "
                "FROM chat_logs ORDER BY id DESC LIMIT %s",
                (max(1, min(limit, 500)),),
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        logfire.warning(f"⚠️ chat_logs read failed: {e}")
        return []
    finally:
        conn.close()


# ── ingest_jobs ───────────────────────────────────────────────────────────────


def record_ingest_started(kind: str) -> int | None:
    conn = _conn()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_jobs (kind, status) VALUES (%s, 'started') RETURNING id",
                (kind,),
            )
            row = cur.fetchone()
        conn.commit()
        return row[0] if row else None
    except Exception as e:
        logfire.warning(f"⚠️ ingest_job start insert failed: {e}")
        return None
    finally:
        conn.close()


def record_ingest_finished(job_id: int | None, status: str, message: str = "") -> None:
    if job_id is None:
        return
    conn = _conn()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ingest_jobs SET status = %s, message = %s, finished_at = now() WHERE id = %s",
                (status, (message or "")[:2000], job_id),
            )
        conn.commit()
    except Exception as e:
        logfire.warning(f"⚠️ ingest_job finish update failed: {e}")
    finally:
        conn.close()


def list_ingest_jobs(limit: int = 20) -> list[dict[str, Any]]:
    conn = _conn()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, kind, status, message, started_at, finished_at "
                "FROM ingest_jobs ORDER BY id DESC LIMIT %s",
                (max(1, min(limit, 200)),),
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        logfire.warning(f"⚠️ ingest_jobs read failed: {e}")
        return []
    finally:
        conn.close()


# ── app_settings ─────────────────────────────────────────────────────────────


def get_setting(key: str, default: str = "") -> str:
    conn = _conn()
    if conn is None:
        return default
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM app_settings WHERE key = %s", (key,))
            row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default
    finally:
        conn.close()


def set_setting(key: str, value: str) -> None:
    conn = _conn()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO app_settings (key, value, updated_at)
                   VALUES (%s, %s, now())
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()""",
                (key, value),
            )
        conn.commit()
    except Exception as e:
        logfire.warning(f"⚠️ app_settings write failed: {e}")
    finally:
        conn.close()


def count_chat_logs(hours: int | None = None) -> int:
    conn = _conn()
    if conn is None:
        return 0
    try:
        with conn.cursor() as cur:
            if hours:
                cur.execute(
                    "SELECT count(*) FROM chat_logs WHERE created_at > now() - make_interval(hours => %s)",
                    (hours,),
                )
            else:
                cur.execute("SELECT count(*) FROM chat_logs")
            row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0
    finally:
        conn.close()


__all__ = [
    "count_chat_logs",
    "get_setting",
    "init_db",
    "list_chat_logs",
    "list_ingest_jobs",
    "log_chat",
    "record_ingest_finished",
    "record_ingest_started",
    "set_setting",
]