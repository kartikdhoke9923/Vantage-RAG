# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
import logfire

from app.config import settings

# Logfire v2 EU tokens start with "pylf_v2_eu_" and must send spans to the
# EU endpoint. If no base URL is configured, infer it from the token prefix
# so the same .env works locally and inside Docker without manual overrides.
_logfire_base_url = settings.LOGFIRE_BASE_URL
if not _logfire_base_url and settings.LOGFIRE_TOKEN:
    if settings.LOGFIRE_TOKEN.startswith("pylf_v2_eu_"):
        _logfire_base_url = "https://logfire-eu.pydantic.dev"

logfire.configure(
    token=settings.LOGFIRE_TOKEN,
    advanced=logfire.AdvancedOptions(base_url=_logfire_base_url) if _logfire_base_url else None,
)

# Capture Jina/Portkey/OpenAI HTTP traffic as structured spans in Logfire.
# instrument_openai tracks the OpenAI-SDK calls that are routed through the
# Portkey gateway (responder, planner); instrument_requests tracks Jina
# embedding/reranking calls made via the requests library.
logfire.instrument_requests()
logfire.instrument_openai()

# Now safe to import app modules - logfire is already active
import json
import os
import time
import traceback
import uuid
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from app import db
from app.agents.graph import build_graph
from app.admin.routes import router as admin_router
from app.guardrails import guard, initialize_rails
from app.health import router as health_router
from app.ingestion.scheduler import start_scheduler
from app.logging import set_request_id
from app.services.health.connection_checker import check_all_connections, log_connection_summary

# Custom Prometheus metrics
RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total number of /query requests",
    ["status"],
)

RAG_REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "Latency of /query requests in seconds",
)

GUARDRAILS_BLOCKS_TOTAL = Counter(
    "guardrails_blocks_total",
    "Number of requests blocked or allowed by guardrails",
    ["blocked"],
)

_security = HTTPBearer(auto_error=False)


def _init_rate_limiter(redis_healthy: bool):
    """Initialize rate limiting. Use Redis in production; fall back to in-memory storage locally."""
    if not settings.RATE_LIMIT_ENABLED:
        logfire.info("Rate limiting disabled (RATE_LIMIT_ENABLED=false) — /query is unthrottled.")
        return False

    from limits.storage import RedisStorage
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.extension import _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address

    if not redis_healthy:
        # Reuse the startup probe from check_all_connections — no second ping.
        app.state.limiter = Limiter(key_func=get_remote_address)
        app.state.rate_limiter_storage = "memory"
        logfire.info("🚦 Rate limiting: Redis unreachable — using in-memory storage.")
    else:
        try:
            storage = RedisStorage(settings.redis_url)
            storage.storage.socket_connect_timeout = 2
            storage.storage.socket_timeout = 2
            if not storage.check() or not storage.storage.ping():
                raise ConnectionError("Redis did not respond to ping")
            app.state.limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)
            app.state.rate_limiter_storage = "redis"
            logfire.info("🚦 Rate limiting initialized via Redis.")
        except Exception as e:
            app.state.limiter = Limiter(key_func=get_remote_address)
            app.state.rate_limiter_storage = "memory"
            logfire.warning(f"⚠️ Redis probed OK but rate limiter setup failed ({e}); using in-memory storage.")

    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return True


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(_security)):
    """
    Require a valid bearer token when RAG_API_KEY is configured.
    In development, omit RAG_API_KEY to disable authentication.
    """
    if not settings.API_KEY:
        # Development mode: no API key required.
        return None

    if not credentials or credentials.credentials != settings.API_KEY:
        logfire.warning("🔒 Unauthorized /query request: invalid or missing API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials



def _get_limiter_rule(times: int, seconds: int) -> str:
    """Convert times/seconds into a slowapi limit string, e.g. '20/minute'."""
    if seconds % 60 == 0:
        return f"{times}/{seconds // 60}minute"
    if seconds % 3600 == 0:
        return f"{times}/{seconds // 3600}hour"
    return f"{times}/{seconds}second"


class _AppLimiter:
    """
    Thin wrapper around the Limiter instance that is initialized at startup.
    Allows routes to be decorated at import time while the real limiter
    (Redis-backed or in-memory) is configured in startup_event.
    """

    def limit(self, rule_or_callable):
        def decorator(func):
            import functools

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                limiter = getattr(app.state, "limiter", None)
                if limiter is None:
                    return func(*args, **kwargs)

                rule = rule_or_callable() if callable(rule_or_callable) else rule_or_callable
                # Build the slowapi wrapper at request time so the limiter
                # instance and storage backend are always current.
                return limiter.limit(rule)(func)(*args, **kwargs)

            return wrapper

        return decorator


app_limiter = _AppLimiter()


def rate_limit(times: int = None, seconds: int = None):
    """
    Decorator factory that applies slowapi rate limiting using the limiter
    initialized at startup. Falls back to a no-op if the limiter is missing.
    The rule is resolved at request time so settings can be overridden in tests.
    """

    def _resolve_rule() -> str:
        t = times or settings.RATE_LIMIT_PER_MINUTE
        s = seconds or 60
        return _get_limiter_rule(t, s)

    return app_limiter.limit(_resolve_rule)


# Initialize FastAPI
app = FastAPI(title="Vantage RAG API")
app.include_router(health_router)
app.include_router(admin_router)

# Cross-origin access for the chat widget + admin page on the portfolio site.
_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    # Explicit lists, not ["*"]: Starlette compares the preflight method/headers
    # literally, so a "*" makes every OPTIONS preflight fail with 400.
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Serve the embeddable chat widget (loader script + styles) as static files.
_WIDGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "widget")
if os.path.isdir(_WIDGET_DIR):
    app.mount("/widget", StaticFiles(directory=_WIDGET_DIR), name="widget")


class _NoCacheWidgetMiddleware:
    """Drop Cache-Control for /widget assets so widget fixes reach browsers on the
    next reload instead of lingering in caches (browsers + Cloudflare)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").startswith("/widget/"):
            async def send_with_cache(message):
                if message["type"] == "http.response.start":
                    headers = [
                        h for h in message.get("headers", []) if h[0].lower() != b"cache-control"
                    ]
                    headers.append((b"cache-control", b"no-cache, no-store, max-age=0"))
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, send_with_cache)
        else:
            await self.app(scope, receive, send)


if os.path.isdir(_WIDGET_DIR):
    app.add_middleware(_NoCacheWidgetMiddleware)

# Expose Prometheus metrics at /metrics with default request instrumentation.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.on_event("startup")
def startup_event():
    initialize_rails()
    db.init_db()

    # Verify all external dependencies are reachable (single probe pass; the
    # cached Redis result is reused by the rate limiter below).
    connection_results = check_all_connections()

    # Build the agent graph with the production checkpointer (Postgres by default).
    app.state.rag_agent = build_graph()

    app.state.rate_limiter_enabled = _init_rate_limiter(
        redis_healthy=connection_results["redis"].healthy
    )

    # Daily re-ingestion tick (reads app_settings.cron_enabled from Neon).
    start_scheduler()

    all_healthy = log_connection_summary(connection_results)
    if settings.STRICT_STARTUP and not all_healthy:
        failed = [name for name, r in connection_results.items() if not r.healthy]
        raise RuntimeError(f"STRICT_STARTUP enabled; failing services: {', '.join(failed)}")

    if not settings.API_KEY:
        logfire.warning("🔓 RAG_API_KEY is not set — /query is open to anyone. Set it in production.")


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    # Per-request Portkey gateway overrides from the UI (fall back to a
    # different provider slug/model). Empty/None → settings defaults.
    slug: Optional[str] = None
    model: Optional[str] = None


@app.get("/")
def home():
    return {"message": "Vantage RAG API is live."}


@app.get("/graph")
def get_graph_image(_api_key: str = Depends(verify_api_key)):
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = app.state.rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


def _run_gate(
    q: str,
    request_id: str,
    thread_id: str,
    start: float,
) -> tuple[str, dict | None]:
    """
    Run the guardrails gate. Returns (decision, payload):
      ("allow", None)            → proceed to the graph
      ("blocked", dict)          → block response to return to the user
      ("unavailable", dict)      → 503 JSON (fail-closed)
    """
    try:
        rail_fired, rail_response = guard(q)
    except Exception as e:
        logfire.error(f"🛡️ Guardrails error: {e}", request_id=request_id, thread_id=thread_id)
        if settings.GUARDRAILS_FAIL_OPEN:
            rail_fired, rail_response = False, None
            logfire.warning("🛡️ Fail-open: proceeding to RAG without gate.")
        else:
            RAG_REQUESTS_TOTAL.labels(status="blocked").inc()
            RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
            return (
                "unavailable",
                JSONResponse(
                    status_code=503,
                    content={
                        "request_id": request_id,
                        "status": "error",
                        "message": "Guardrails unavailable. Please try again later.",
                    },
                ),
            )
    if rail_fired:
        GUARDRAILS_BLOCKS_TOTAL.labels(blocked="true").inc()
        RAG_REQUESTS_TOTAL.labels(status="blocked").inc()
        RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
        logfire.info("🛡️ Request blocked by guardrails", request_id=request_id, thread_id=thread_id)
        return (
            "blocked",
            {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
            },
        )

    GUARDRAILS_BLOCKS_TOTAL.labels(blocked="false").inc()
    return "allow", None


def _build_initial_state(q: str, thread_id: str, body) -> dict:
    """Shared initial LangGraph state for the query endpoints."""
    return {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "slug": body.slug,
        "model": body.model,
    }


def _response_from_state(q: str, final_output: dict) -> dict:
    """Shape a full /query response from a finalized graph state."""
    return {
        "question": q,
        "answer": final_output.get("final_answer"),
        "thought_process": final_output.get("plan"),
        "status": final_output.get("status"),
        "sources": final_output.get("documents", []),
        "fact_check": final_output.get("fact_check"),
        "citation_warning": final_output.get("citation_warning"),
    }


def _log_exchange(q: str, thread_id: str, start: float, final_output: dict | None, cache_hit: str = "none"):
    """Best-effort row into chat_logs so the admin page can review conversations."""
    try:
        db.log_chat(
            thread_id=thread_id,
            question=q,
            answer=final_output.get("final_answer") if final_output else "",
            status=final_output.get("status") or "error" if final_output else "error",
            latency_ms=int((time.perf_counter() - start) * 1000),
            cache_hit=cache_hit,
            source_count=len(final_output.get("documents") or []) if final_output else 0,
        )
    except Exception:  # noqa: BLE001 - logging must never break a response
        pass


@app.post("/query")
@rate_limit()
def query(
    request: Request,
    body: QueryRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Runs the LangGraph RAG pipeline synchronously.
    Returns the final answer, thought process, status, and sources.
    """
    q = body.q
    thread_id = body.thread_id
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    start = time.perf_counter()
    with logfire.span("🔍 /query", request_id=request_id, thread_id=thread_id):
        # Gate: run guardrails synchronously so blocked requests never run the graph.
        decision, payload = _run_gate(q, request_id, thread_id, start)
        if decision != "allow":
            return payload

        try:
            rag_agent = app.state.rag_agent
            initial_state = _build_initial_state(q, thread_id, body)
            config = {"configurable": {"thread_id": thread_id}}
            final_output = rag_agent.invoke(initial_state, config=config)

            RAG_REQUESTS_TOTAL.labels(status="success").inc()
            RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
            logfire.info(
                "✅ RAG pipeline completed",
                request_id=request_id,
                thread_id=thread_id,
            )
            response = _response_from_state(q, final_output)
            _log_exchange(q, thread_id, start, final_output)
            return response
        except Exception as e:
            RAG_REQUESTS_TOTAL.labels(status="error").inc()
            RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
            tb = traceback.format_exc()
            print(f"❌ RAG pipeline failed: {e}\n{tb}", flush=True)
            logfire.error(
                f"❌ RAG pipeline failed: {e}\n{tb}",
                request_id=request_id,
                thread_id=thread_id,
            )
            _log_exchange(q, thread_id, start, None)
            return JSONResponse(
                status_code=500,
                content={
                    "request_id": request_id,
                    "status": "error",
                    "message": f"Failed to process request: {e}",
                },
            )


@app.post("/query/stream")
@rate_limit()
def query_stream(
    request: Request,
    body: QueryRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Runs the LangGraph RAG pipeline as a Server-Sent-Events stream.

    Emits one SSE event per node visit, then a final 'done' event with the
    complete answer. Event shape: {"type": "start"|"node"|"done"|"error", ...}.
    """
    q = body.q
    thread_id = body.thread_id
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    start = time.perf_counter()
    with logfire.span("🔍 /query/stream", request_id=request_id, thread_id=thread_id):
        rag_agent = app.state.rag_agent

        def sse(data: dict) -> str:
            return f"data: {json.dumps(data, default=str)}\n\n"

        def event_stream():
            try:
                # Emit the start event BEFORE the gate so the client gets bytes
                # immediately and no request ever sits at 0 bytes.
                yield sse({"type": "start", "question": q})

                decision, payload = _run_gate(q, request_id, thread_id, start)
                if decision != "allow":
                    if isinstance(payload, JSONResponse):
                        try:
                            msg = json.loads(payload.body).get("message", "Request blocked.")
                        except Exception:
                            msg = "Request blocked."
                        yield sse({"type": "error", "message": msg})
                    else:
                        yield sse({"type": "done", "answer": payload.get("answer", "Blocked."),
                                   "sources": [], "thought_process": [], "status": "Blocked."})
                    return

                initial_state = _build_initial_state(q, thread_id, body)
                config = {"configurable": {"thread_id": thread_id}}

                for chunk in rag_agent.stream(initial_state, config, stream_mode="updates"):
                    for node_name, update in chunk.items():
                        yield sse(
                            {
                                "type": "node",
                                "node": node_name,
                                "status": update.get("status"),
                                "plan": update.get("plan"),
                                "fact_check": update.get("fact_check"),
                                "intent": update.get("intent"),
                            }
                        )

                final_state = rag_agent.get_state(config).values if hasattr(rag_agent, "get_state") else {}
                RAG_REQUESTS_TOTAL.labels(status="success").inc()
                RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
                logfire.info(
                    "✅ RAG pipeline streamed",
                    request_id=request_id,
                    thread_id=thread_id,
                )
                _log_exchange(q, thread_id, start, final_state)
                yield sse(
                    {
                        "type": "done",
                        "answer": final_state.get("final_answer"),
                        "sources": final_state.get("documents", []),
                        "thought_process": final_state.get("plan"),
                        "status": final_state.get("status"),
                        "fact_check": final_state.get("fact_check"),
                        "citation_warning": final_state.get("citation_warning"),
                    }
                )
            except Exception as e:
                RAG_REQUESTS_TOTAL.labels(status="error").inc()
                RAG_REQUEST_DURATION.observe(time.perf_counter() - start)
                tb = traceback.format_exc()
                print(f"❌ RAG stream failed: {e}\n{tb}", flush=True)
                logfire.error(
                    f"❌ RAG stream failed: {e}\n{tb}",
                    request_id=request_id,
                    thread_id=thread_id,
                )
                _log_exchange(q, thread_id, start, None)
                yield sse({"type": "error", "message": f"Failed to process request: {e}"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
