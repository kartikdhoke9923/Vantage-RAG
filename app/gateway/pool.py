"""3-model rotation pool for the LLM gateway.

Each role (guardrails / planner / responder) has a fixed model from env.
When a model fails it goes on a short cool-down and the caller rotates to
the next model in the pool instead of hammering the same rate-limited endpoint.
"""
import threading
import time

from app.config import settings

# 429 / rate-limit → back off longer; other errors → short cool-down.
RATE_LIMIT_COOLDOWN = 120.0
FAIL_COOLDOWN = 30.0

_lock = threading.Lock()
_cooldowns: dict[str, float] = {}


def _pool() -> list[str]:
    """Deduplicated set of the three role models, in guardrail/planner/responder order."""
    models = [
        settings.PORTKEY_MODEL_GUARDRAIL,
        settings.PORTKEY_MODEL_PLANNER,
        settings.PORTKEY_MODEL_RESPONDER,
    ]
    return list(dict.fromkeys(m for m in models if m))


ROLE_TO_MODEL = {
    "guardrails": lambda: settings.PORTKEY_MODEL_GUARDRAIL,
    "planner": lambda: settings.PORTKEY_MODEL_PLANNER,
    "responder": lambda: settings.PORTKEY_MODEL_RESPONDER,
    # unassigned roles fall back to the responder model
    "researcher": lambda: settings.PORTKEY_MODEL_RESPONDER,
    "analyst": lambda: settings.PORTKEY_MODEL_RESPONDER,
    "coder": lambda: settings.PORTKEY_MODEL_RESPONDER,
    "fact_checker": lambda: settings.PORTKEY_MODEL_RESPONDER,
}


def role_model(feature: str) -> str:
    """Primary routing string for a feature; falls back to the gateway default."""
    getter = ROLE_TO_MODEL.get(feature)
    if getter:
        routing = getter()
        if routing:
            return routing
    from app.gateway.client import gateway_model

    return gateway_model()


def routing_parts(routing: str) -> tuple[str | None, str | None]:
    """'@slug/model' → (slug, model)."""
    if routing.startswith("@"):
        routing = routing[1:]
    if "/" in routing:
        slug, model = routing.split("/", 1)
        return slug, model
    return None, routing


def mark_failure(routing: str, rate_limit: bool = False) -> None:
    with _lock:
        _cooldowns[routing] = time.monotonic() + (
            RATE_LIMIT_COOLDOWN if rate_limit else FAIL_COOLDOWN
        )


def mark_success(routing: str) -> None:
    with _lock:
        _cooldowns.pop(routing, None)


def pick_candidates(feature: str, slug: str | None = None, model: str | None = None) -> list[str]:
    """Ordered candidate routing strings: the role's fixed model first, then
    the rest of the pool, then the per-request slug/model override (escape
    hatch). Models on cool-down are skipped; if every model is cooling down we
    clear once so the request can still go out."""
    from app.gateway.client import gateway_model

    base = role_model(feature)

    picks: list[str] = []
    if base not in picks:
        picks.append(base)
    for m in _pool():
        if m not in picks:
            picks.append(m)
    if slug or model:
        override = (
            f"@{slug or settings.PORTKEY_PRIMARY_SLUG}/{model or settings.PORTKEY_PRIMARY_MODEL}"
        )
        if override not in picks:
            picks.append(override)
    if not picks:
        picks.append(gateway_model())

    with _lock:
        now = time.monotonic()
        active = [m for m in picks if _cooldowns.get(m, 0.0) <= now]
        if not active:
            _cooldowns.clear()
            active = picks
    return active


__all__ = [
    "FAIL_COOLDOWN",
    "RATE_LIMIT_COOLDOWN",
    "mark_failure",
    "mark_success",
    "pick_candidates",
    "role_model",
    "routing_parts",
]