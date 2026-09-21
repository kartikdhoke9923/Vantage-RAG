import logfire
from langchain_openai import ChatOpenAI
from nemoguardrails import LLMRails, RailsConfig
from openai import RateLimitError
from portkey_ai import PORTKEY_GATEWAY_URL

from app.config import settings
from app.gateway import pool
from app.gateway.client import _api_key, _make_headers
from app.guardrails.colang_rules import COLANG_CONTENT, RAIL_INDICATORS, YAML_CONTENT

_rails: LLMRails | None = None

# Cap the gate's output so a single NeMo rail call can never request huge
# generations (Groq qwen-style OTPM limits reject requests > 1k output tokens).
GATE_MAX_TOKENS = 500


class _RotatingGuardLLM(ChatOpenAI):
    """ChatOpenAI that hops to the next pool model when the primary errors, so a
    rate-limited rail model swaps instead of failing the gate."""

    def __init__(self, llms: list[ChatOpenAI], **primary_kwargs):
        super().__init__(**primary_kwargs)
        self._pool_llms = llms

    def _mark(self, llm, e: Exception) -> None:
        routing = getattr(llm, "_routing", None)
        if routing:
            pool.mark_failure(routing, rate_limit=isinstance(e, RateLimitError))

    def generate(self, messages, stop=None, callbacks=None, *, tags=None, metadata=None, **kwargs):
        last_error: Exception | None = None
        for llm in self._pool_llms:
            try:
                return llm.generate(
                    messages, stop=stop, callbacks=callbacks, tags=tags, metadata=metadata, **kwargs
                )
            except Exception as e:
                last_error = e
                self._mark(llm, e)
        raise last_error or RuntimeError("Guardrail LLM chain exhausted.")

    async def agenerate(self, messages, stop=None, callbacks=None, *, tags=None, metadata=None, **kwargs):
        last_error: Exception | None = None
        for llm in self._pool_llms:
            try:
                return await llm.agenerate(
                    messages, stop=stop, callbacks=callbacks, tags=tags, metadata=metadata, **kwargs
                )
            except Exception as e:
                last_error = e
                self._mark(llm, e)
        raise last_error or RuntimeError("Guardrail LLM chain exhausted.")


def _build_guard_llm() -> ChatOpenAI | None:
    """
    Pick the gate LLM, preferring the Portkey gateway so one slug governs both
    the RAG pipeline and the rails (e.g. GEMINI via PORTKEY_PRIMARY_SLUG):

      1. OPENAI_API_KEY set → OpenAI gpt-5-mini (paid).
      2. PORTKEY_API_KEY set → gateway @<slug>/<model>, rotating across the
         whole model pool so a rate-limited guardrail model swaps to the next.
      3. GROQ_API_KEY set    → free Groq endpoint (GUARDRAIL_MODEL).
      4. None of the above  → guardrails disabled; messages pass through to RAG.
    """
    llm_kwargs: dict = {"max_tokens": GATE_MAX_TOKENS}

    if settings.OPENAI_API_KEY:
        llm_kwargs["api_key"] = settings.OPENAI_API_KEY
        llm_kwargs["model"] = settings.GUARDRAIL_MODEL or "gpt-5-mini"
    elif settings.PORTKEY_API_KEY:
        candidates = pool.pick_candidates("guardrails")
        llms = [
            ChatOpenAI(
                api_key=_api_key(),
                base_url=PORTKEY_GATEWAY_URL,
                model=routing,
                default_headers=_make_headers("guardrails"),
                **llm_kwargs,
            )
            for routing in candidates
        ]
        for llm, routing in zip(llms, candidates):
            llm._routing = routing
        guard_llm = _RotatingGuardLLM(
            llms,
            api_key=_api_key(),
            base_url=PORTKEY_GATEWAY_URL,
            model=candidates[0],
            default_headers=_make_headers("guardrails"),
            **llm_kwargs,
        )
        return guard_llm
    elif settings.GROQ_API_KEY:
        llm_kwargs["api_key"] = settings.GROQ_API_KEY
        llm_kwargs["base_url"] = "https://api.groq.com/openai/v1"
        llm_kwargs["model"] = settings.GUARDRAIL_MODEL or "openai/gpt-oss-20b"
    else:
        return None

    return ChatOpenAI(**llm_kwargs)


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    """
    global _rails

    guard_llm = _build_guard_llm()
    if guard_llm is None:
        _rails = None
        logfire.warning(
            "No OPENAI_API_KEY / PORTKEY_API_KEY / GROQ_API_KEY set — NeMo "
            "guardrails disabled; messages pass through to RAG."
        )
        return

    config = RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT)

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised (pool: {pool.role_model('guardrails')}).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
