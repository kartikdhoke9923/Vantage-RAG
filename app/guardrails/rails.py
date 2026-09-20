import logfire
from langchain_openai import ChatOpenAI
from nemoguardrails import LLMRails, RailsConfig
from portkey_ai import PORTKEY_GATEWAY_URL

from app.config import settings
from app.gateway.client import _api_key, _make_headers, gateway_model
from app.guardrails.colang_rules import COLANG_CONTENT, RAIL_INDICATORS, YAML_CONTENT

_rails: LLMRails | None = None

# Cap the gate's output so a single NeMo rail call can never request huge
# generations (Groq qwen-style OTPM limits reject requests > 1k output tokens).
GATE_MAX_TOKENS = 500


def _build_guard_llm() -> ChatOpenAI | None:
    """
    Pick the gate LLM, preferring the Portkey gateway so one slug governs both
    the RAG pipeline and the rails (e.g. GEMINI via PORTKEY_PRIMARY_SLUG):

      1. OPENAI_API_KEY set → OpenAI gpt-5-mini (paid).
      2. PORTKEY_API_KEY set → gateway @<slug>/<model>.
      3. GROQ_API_KEY set    → free Groq endpoint (GUARDRAIL_MODEL).
      4. None of the above  → guardrails disabled; messages pass through to RAG.
    """
    llm_kwargs: dict = {"max_tokens": GATE_MAX_TOKENS}

    if settings.OPENAI_API_KEY:
        llm_kwargs["api_key"] = settings.OPENAI_API_KEY
        llm_kwargs["model"] = settings.GUARDRAIL_MODEL or "gpt-5-mini"
    elif settings.PORTKEY_API_KEY:
        llm_kwargs["api_key"] = _api_key()
        llm_kwargs["base_url"] = PORTKEY_GATEWAY_URL
        llm_kwargs["model"] = gateway_model()
        llm_kwargs["default_headers"] = _make_headers("guardrails")
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
    logfire.info(f"🛡️ NeMo Guardrails initialised ({guard_llm.model_name}).")


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
