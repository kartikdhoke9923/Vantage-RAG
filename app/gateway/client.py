import logfire
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings
from app.gateway import cache
from app.gateway import metrics as gateway_metrics

# Portkey routing strategy:
#   - Primary/fallback logic lives in a Portkey saved config referenced via the
#     x-portkey-config-id header (block_inline_config is enabled on this account).
#   - If no config ID is set, routing falls back to the `@<slug>/<model>` form in
#     the request — only PORTKEY_API_KEY is required to boot the app.
#   - The inline config dict approach is disabled for this account, so all
#     retry/fallback/cache behavior must be configured inside the Portkey UI.

_warned_missing_key = False


def _make_headers(feature: str = "rag") -> dict:
    """Build Portkey headers.

    Only PORTKEY_API_KEY is required; missing credentials must never block
    import — actual requests will fail loudly from the gateway instead.
    When PORTKEY_PRIMARY_CONFIG_ID is absent, routing falls back to the
    @<slug>/<model> path passed in the request.
    """
    global _warned_missing_key
    if not settings.PORTKEY_API_KEY:
        if not _warned_missing_key:
            _warned_missing_key = True
            logfire.warning(
                "PORTKEY_API_KEY not set — LLM gateway calls will fail "
                "until it is configured."
            )
        return {"x-portkey-api-key": ""}
    kwargs: dict = {
        "api_key": settings.PORTKEY_API_KEY,
        "metadata": {
            "feature": feature,
            "_user": "rag-system",
            "environment": "production",
        },
    }
    if settings.PORTKEY_PRIMARY_CONFIG_ID:
        kwargs["config_id"] = settings.PORTKEY_PRIMARY_CONFIG_ID
    return createHeaders(**kwargs)


# OpenAI-compatible client routed through Portkey.
# We use the OpenAI SDK directly because the native Portkey SDK does not
# surface a first-class config_id constructor parameter; the header-based
# approach works reliably with block_inline_config enabled.

# Fall back to a placeholder key so the app can import without credentials;
# real requests will fail loudly from the gateway until PORTKEY_API_KEY is set.
def _api_key() -> str:
    return settings.PORTKEY_API_KEY or "portkey-not-configured"


portkey_client = OpenAI(
    api_key=_api_key(),
    base_url=PORTKEY_GATEWAY_URL,
    default_headers=_make_headers(),
)


def gateway_model(slug: str | None = None, model: str | None = None) -> str:
    """Portkey routing string `@<slug>/<model>` with per-request overrides."""
    return f"@{slug or settings.PORTKEY_PRIMARY_SLUG}/{model or settings.PORTKEY_PRIMARY_MODEL}"


def get_langchain_llm(feature: str = "rag", slug: str | None = None, model: str | None = None) -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI - a drop-in for LangChain nodes.

    `slug`/`model` override settings when provided (used by the UI gateway
    switcher to fall back to a different provider on the fly).

    Why ChatOpenAI:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + saved-config reference). The @slug/model-name format is Portkey-specific - the
      upstream provider's own client does not understand it. Portkey is just in the middle.
    """
    return ChatOpenAI(
        api_key=_api_key(),
        base_url=PORTKEY_GATEWAY_URL,
        model=gateway_model(slug, model),
        default_headers=_make_headers(feature),
    )


def get_async_openai_client(feature: str = "rag") -> AsyncOpenAI:
    """
    Returns an async OpenAI client that routes through the Portkey gateway.
    Use this for non-LangChain async LLM calls (e.g. async FastAPI endpoints).
    """
    return AsyncOpenAI(
        api_key=_api_key(),
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=_make_headers(feature),
    )


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the response.

    The OpenAI SDK does not expose raw headers on parsed responses, so cache
    hit/miss tracking is best-effort. We inspect common attribute paths and
    fall back to 'MISS'.
    """
    for attr in ("_raw_response", "_response", "_http_response", "headers"):
        raw = getattr(response, attr, None)
        if raw is not None:
            headers = getattr(raw, "headers", None)
            if headers is not None:
                status = headers.get("x-portkey-cache-status", "")
                if status:
                    return status.upper()
    return "MISS"


# ---------------------------------------------------------------------------
# Fallback chain + usage + cost
# ---------------------------------------------------------------------------

def _fallback_groq_llm() -> ChatOpenAI | None:
    """Direct-to-Groq ChatOpenAI used when the Portkey gateway fails.

    Uses GROQ_FALLBACK_API_KEY first, then GROQ_API_KEY. Returns None when no
    Groq key is configured (the chain then fails loudly = gate-off).
    """
    api_key = settings.GROQ_FALLBACK_API_KEY or settings.GROQ_API_KEY
    if not api_key:
        return None
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model=settings.GUARDRAIL_MODEL or "openai/gpt-oss-20b",
    )


def _fallback_groq_model() -> str:
    """Model used by the direct-Groq fallback path."""
    return settings.GUARDRAIL_MODEL or "openai/gpt-oss-20b"


def _fallback_groq_client() -> "OpenAI | None":
    """OpenAI SDK client pointed straight at Groq (same key/model as _fallback_groq_llm).

    Returns None when no Groq key is configured (the chain then fails loudly = gate-off).
    """
    api_key = settings.GROQ_FALLBACK_API_KEY or settings.GROQ_API_KEY
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def _message_usage(message) -> tuple[int, int]:
    """Best-effort (prompt_tokens, completion_tokens) from a ChatOpenAI result."""
    metadata = getattr(message, "usage_metadata", None) or {}
    if metadata:
        return int(metadata.get("prompt_tokens") or 0), int(metadata.get("completion_tokens") or 0)
    response_metadata = getattr(message, "response_metadata", None) or {}
    usage = response_metadata.get("token_usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def invoke_llm_with_fallback(
    prompt: str,
    feature: str = "rag",
    slug: str | None = None,
    model: str | None = None,
) -> str:
    """
    Invoke the LLM with an in-code fallback chain and optional Redis caching.

    Chain order:
      1. Portkey gateway with the per-request slug/model override.
      2. Portkey gateway with settings defaults (covers a bad UI override).
      3. Direct Groq fallback key → openai/gpt-oss-20b.
      4. Gate-off: re-raise the last error (never silently swallow failures).

    Each successful call also records tokens + approx cost as Prometheus
    metrics (see gateway/metrics.py). Responses are cached in Redis when the
    feature/model/prompt triple has been served before.
    """
    routing_model = gateway_model(slug, model)

    def _produce() -> str:
        attempts: list[tuple[str, ChatOpenAI]] = [(gateway_model(slug, model), get_langchain_llm(feature, slug, model))]
        if slug or model:
            attempts.append((gateway_model(), get_langchain_llm(feature)))
        fallback = _fallback_groq_llm()
        if fallback is not None:
            attempts.append(("groq-fallback", fallback))

        last_error: Exception | None = None
        for status_label, llm in attempts:
            try:
                message = llm.invoke(prompt)
                pt, ct = _message_usage(message)
                gateway_metrics.record_usage(feature, status_label, pt, ct)
                gateway_metrics.GATEWAY_CALLS_TOTAL.labels(feature=feature, status=status_label).inc()
                logfire.info(
                    f"🔁 Gateway '{status_label}' ok ({pt} prompt / {ct} completion tokens)."
                )
                return message.content
            except Exception as e:  # noqa: BLE001 - fallback chain requires a blind catch
                last_error = e
                logfire.error(f"⚠️ Gateway '{status_label}' failed: {e}")
                gateway_metrics.GATEWAY_CALLS_TOTAL.labels(feature=feature, status="error").inc()

        raise RuntimeError(f"LLM gateway unavailable for '{feature}': {last_error}")

    return cache.cached_completion(feature, routing_model, prompt, _produce)
