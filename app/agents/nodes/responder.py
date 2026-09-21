import re

import logfire
from openai import RateLimitError
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.agents.state import AgentState
from app.config import settings
from app.gateway import cache as gateway_cache
from app.gateway import extract_cache_status, gateway_model, portkey_client
from app.gateway import metrics as gateway_metrics
from app.gateway import pool
from app.safety.pii import mask_pii
from app.services.prompts import render_prompt

# Max characters of retrieved context passed to the LLM.
MAX_CONTEXT_CHARS = 25000
CITATION_RE = re.compile(r"\[(\d+)\]")


def _build_numbered_context(documents: list[dict]) -> str:
    """
    Format retrieval chunks with citation markers so the LLM can cite them:

        [1] (source: architecture.pptx) <content>
        [2] (source: cronjobs.docx)     <content>
    """
    blocks = []
    for index, doc in enumerate(documents, start=1):
        source = doc.get("source", "unknown")
        content = doc.get("content", "")
        source_line = f"[{index}] (source: {source})"
        blocks.append(f"{source_line}\n{content}")
    return "\n\n".join(blocks)


def _check_citations(content: str, max_index: int) -> str | None:
    """
    Citation guard: every [n] cited in the answer must refer to one of the
    retrieved chunks ([1..max_index]). Returns a warning string, or None when
    the answer only cites valid sources (or cites nothing).
    """
    cited = [int(n) for n in CITATION_RE.findall(content)]
    invalid = [n for n in cited if n < 1 or n > max_index]
    if invalid:
        message = f"Citation guard: answer referenced sources {sorted(set(invalid))} outside the retrieved chunk range 1..{max_index}."
        logfire.warning(message)
        return message
    return None


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation
    History, with enforced inline citations for retrieved chunks.

    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = render_prompt(
            "responder_conversational",
            history=history_str,
            question=user_msg,
        )
    else:
        logfire.info("Generating technical RAG response with citations.")
        documents = state.get("documents", [])
        tool_results = state.get("tool_results") or ""

        if state.get("intent") == "tool" and tool_results and not documents:
            numbered_context = tool_results
            max_index = "0"
        else:
            numbered_context = _build_numbered_context(documents)
            max_index = str(len(documents))

        prompt = render_prompt(
            "responder_technical",
            context=numbered_context,
            history=history_str,
            question=user_msg,
            max_index=max_index,
            analysis=state.get("analysis") or "",
        )

    with logfire.span("✍️ LLM Synthesis"):
        try:
            content, cache_status = _generate_response_cached(
                prompt, state.get("slug"), state.get("model")
            )
            is_cache_hit = cache_status in ("HIT", "REDIS")

            # Output PII / secret masking (opt-in via MASK_PII_IN_OUTPUT).
            if settings.MASK_PII_IN_OUTPUT and content:
                masked, findings = mask_pii(content)
                if findings:
                    kinds = sorted({kind for kind, _ in findings})
                    logfire.warning(f"🔒 Output PII masked: {', '.join(kinds)}")
                    plan_update_hint = ", ".join(kinds)
                content = masked
            else:
                plan_update_hint = None

            # Citation guard: verify every cited source is a real retrieved chunk.
            citation_warning = None
            if query != "CONVERSATIONAL":
                citation_warning = _check_citations(content, len(state.get("documents", [])))

            if is_cache_hit:
                logfire.info(f"⚡ Gateway Cache {cache_status} — response served from cache.")
                plan_update = state["plan"] + [f"Cache: {cache_status} ⚡"]
                status = f"Cache hit ({cache_status}) — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            if citation_warning:
                plan_update = plan_update + ["⚠️ " + citation_warning]

            if plan_update_hint:
                plan_update = plan_update + [f"🔒 Output masked ({plan_update_hint})"]

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "citation_warning": citation_warning,
                "messages": [{"role": "assistant", "content": content}],
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed after retries: {e}")
            raise


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=3),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _generate_response(prompt: str, slug: str | None = None, model: str | None = None):
    """Call the Portkey LLM gateway, rotating through the model pool on failure."""
    last_error: Exception | None = None
    for routing in pool.pick_candidates("responder", slug, model):
        try:
            response = portkey_client.chat.completions.create(
                model=routing,
                messages=[{"role": "user", "content": prompt}],
            )
            pool.mark_success(routing)
            return response
        except RateLimitError as e:
            last_error = e
            pool.mark_failure(routing, rate_limit=True)
            logfire.warning(f"⚠️ Responder '{routing}' rate-limited: {e}")
        except Exception as e:
            last_error = e
            pool.mark_failure(routing)
            logfire.warning(f"⚠️ Responder '{routing}' failed: {e}")
    raise last_error or RuntimeError("Responder gateway unavailable.")


def _generate_response_fallback(prompt: str, slug: str | None, model: str | None):
    """_generate_response, falling back to a direct Groq call on gateway failure."""
    try:
        return _generate_response(prompt, slug, model)
    except Exception:
        from app.gateway.client import _fallback_groq_client, _fallback_groq_model

        fallback = _fallback_groq_client()
        if fallback is None:
            raise
        logfire.warning("⚠️ Portkey synthesis failed — falling back to direct Groq.")
        return fallback.chat.completions.create(
            model=_fallback_groq_model(),
            messages=[{"role": "user", "content": prompt}],
        )


def _generate_response_cached(prompt: str, slug: str | None, model: str | None) -> tuple[str, str]:
    """
    Redis-cached variant of _generate_response.

    Returns (content, cache_status) where cache_status is 'REDIS' for a Redis
    hit, the Portkey x-portkey-cache-status, or 'MISS'. Also records tokens &
    approximate cost to Prometheus.
    """
    routing = gateway_model(slug, model)
    cached = gateway_cache.cache_get("responder", routing, prompt)
    if cached is not None:
        gateway_metrics.GATEWAY_CACHE_HITS_TOTAL.labels(feature="responder").inc()
        logfire.info("⚡ Redis cache hit (responder)")
        return cached, "REDIS"

    response = _generate_response_fallback(prompt, slug, model)
    content = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    if usage is not None:
        gateway_metrics.record_usage(
            "responder",
            routing,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )
    gateway_metrics.GATEWAY_CALLS_TOTAL.labels(feature="responder", status="portkey").inc()
    if content:
        gateway_cache.cache_put("responder", routing, prompt, content)
    return content, extract_cache_status(response)