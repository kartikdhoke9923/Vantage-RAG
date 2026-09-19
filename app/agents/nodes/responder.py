import re

import logfire
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.agents.state import AgentState
from app.gateway import extract_cache_status, portkey_client
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

        full_context = ""
        for doc in documents:
            if len(full_context) + len(doc.get("content", "")) < MAX_CONTEXT_CHARS:
                full_context += doc.get("content", "") + "\n\n"
            else:
                logfire.warning("Context truncated to fit TPM limits.")
                break

        numbered_context = _build_numbered_context(documents)
        prompt = render_prompt(
            "responder_technical",
            context=numbered_context,
            history=history_str,
            question=user_msg,
            max_index=str(len(documents)),
        )

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = _generate_response(prompt, slug=state.get("slug"), model=state.get("model"))
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            # Citation guard: verify every cited source is a real retrieved chunk.
            citation_warning = None
            if query != "CONVERSATIONAL":
                citation_warning = _check_citations(content, len(state.get("documents", [])))

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            if citation_warning:
                plan_update = plan_update + ["⚠️ " + citation_warning]

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
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _generate_response(prompt: str, slug: str | None = None, model: str | None = None):
    """Call the LLM gateway with retry logic for transient failures."""
    from app.gateway import gateway_model

    return portkey_client.chat.completions.create(
        model=gateway_model(slug, model),
        messages=[{"role": "user", "content": prompt}],
    )