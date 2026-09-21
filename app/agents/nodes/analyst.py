"""Analyst sub-agent — evidence distillation.

Turns the merged retrieval chunks into a concise evidence digest (with [n]
citation markers) that the responder uses as secondary context. Non-fatal: if
the LLM is unavailable, the pipeline continues without analysis.
"""
import logfire

from app.agents.state import AgentState
from app.gateway import invoke_llm_with_fallback
from app.services.prompts import render_prompt

MAX_EVIDENCE_CHARS = 8000


def _numbered_evidence(documents: list[dict]) -> str:
    blocks = []
    used = 0
    for index, doc in enumerate(documents, start=1):
        content = doc.get("content", "")
        remaining = MAX_EVIDENCE_CHARS - used
        if len(content) > remaining:
            content = content[:remaining] + "…"
        blocks.append(
            f"[{index}] (source: {doc.get('source', 'unknown')})\n{content}"
        )
        used += len(content)
        if used >= MAX_EVIDENCE_CHARS:
            break
    return "\n\n".join(blocks)


def analyst_node(state: AgentState):
    """Distill the retrieved evidence into an analysis digest."""
    documents = state.get("documents") or []
    if not documents:
        return {"analysis": None, "plan": state["plan"] + ["Sub-agent: analyst (no evidence)"]}

    evidence = _numbered_evidence(documents)
    question = state["messages"][-1]["content"] if state["messages"] else state["current_query"]
    prompt = render_prompt("analyst", evidence=evidence, question=question)

    with logfire.span("🧠 Analyst Digestion"):
        try:
            analysis = invoke_llm_with_fallback(prompt, feature="analyst")
        except Exception as e:  # noqa: BLE001 - non-fatal by design
            logfire.warning(f"Analyst unavailable ({e}); skipping digest.")
            return {
                "analysis": None,
                "plan": state["plan"] + ["Sub-agent: analyst (skipped: LLM error)"],
            }

    logfire.info("🧠 Analyst produced evidence digest.")
    return {
        "analysis": analysis,
        "plan": state["plan"] + ["Sub-agent: analyst"],
        "status": "Analyst distilled evidence.",
    }