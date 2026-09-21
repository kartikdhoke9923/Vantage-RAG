"""Coder sub-agent — code-guidance extraction.

For coding questions, extracts the code-relevant guidance from the retrieved
evidence (settings, APIs, commands, config patterns with [n] citations) into
`analysis`, which the responder uses to write the final solution.
Non-fatal like the analyst.
"""
import logfire

from app.agents.state import AgentState
from app.gateway import invoke_llm_with_fallback
from app.services.prompts import render_prompt

MAX_EVIDENCE_CHARS = 8000


def coder_node(state: AgentState):
    """Extract code guidance from the evidence into the analysis field."""
    documents = state.get("documents") or []
    if not documents:
        return {"analysis": None, "plan": state["plan"] + ["Sub-agent: coder (no evidence)"]}

    blocks = []
    used = 0
    for index, doc in enumerate(documents, start=1):
        content = doc.get("content", "")
        remaining = MAX_EVIDENCE_CHARS - used
        if len(content) > remaining:
            content = content[:remaining] + "…"
        blocks.append(f"[{index}] (source: {doc.get('source', 'unknown')})\n{content}")
        used += len(content)
        if used >= MAX_EVIDENCE_CHARS:
            break
    evidence = "\n\n".join(blocks)
    question = state["messages"][-1]["content"] if state["messages"] else state["current_query"]
    prompt = render_prompt("coder", evidence=evidence, question=question)

    with logfire.span("💻 Coder Analysis"):
        try:
            guidance = invoke_llm_with_fallback(prompt, feature="coder")
        except Exception as e:  # noqa: BLE001 - non-fatal by design
            logfire.warning(f"Coder unavailable ({e}); skipping code guidance.")
            return {
                "analysis": None,
                "plan": state["plan"] + ["Sub-agent: coder (skipped: LLM error)"],
            }

    logfire.info("💻 Coder produced code guidance.")
    return {
        "analysis": guidance,
        "plan": state["plan"] + ["Sub-agent: coder"],
        "status": "Coder extracted code guidance.",
    }