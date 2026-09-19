import logfire
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings

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
