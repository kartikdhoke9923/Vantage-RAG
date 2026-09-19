import logfire
from langchain_openai import ChatOpenAI
from nemoguardrails import LLMRails, RailsConfig

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, RAIL_INDICATORS, YAML_CONTENT

_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.

    LLM selection (no paid key required):
      1. OPENAI_API_KEY set      → OpenAI gpt-5-mini (paid).
      2. GROQ_API_KEY set        → free Groq endpoint, openai/gpt-oss-20b
                                  (or GUARDRAIL_MODEL if set; the key's free
                                  catalog no longer includes llama-3.1-8b-instant).
      3. Neither set             → guardrails disabled; messages pass through to RAG.
    """
    global _rails

    if settings.OPENAI_API_KEY:
        guard_api_key = settings.OPENAI_API_KEY
        guard_base_url = None  # OpenAI's default endpoint
        guard_model = settings.GUARDRAIL_MODEL or "gpt-5-mini"
    elif settings.GROQ_API_KEY:
        guard_api_key = settings.GROQ_API_KEY
        guard_base_url = "https://api.groq.com/openai/v1"
        guard_model = settings.GUARDRAIL_MODEL or "openai/gpt-oss-20b"
    else:
        _rails = None
        logfire.warning(
            "No OPENAI_API_KEY / GROQ_API_KEY set — NeMo guardrails disabled; "
            "messages pass through to RAG."
        )
        return

    llm_kwargs: dict = {"api_key": guard_api_key, "model": guard_model}
    if guard_base_url:
        llm_kwargs["base_url"] = guard_base_url
    guard_llm = ChatOpenAI(**llm_kwargs)

    config = RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT)

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({guard_model}).")


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
