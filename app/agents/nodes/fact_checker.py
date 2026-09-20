"""Runtime fact-checker node.

Verifies the synthesized answer's claims against the retrieved evidence with a
cheap LLM verdict (GROUNDED / PARTIAL / UNGROUNDED / UNKNOWN). Non-blocking:
problems are surfaced as a caution note appended to the answer, never fatal.

Can be disabled via FACT_CHECK_ENABLED=false; skips automatically when the
request is conversational (no retrieved documents).
"""
import logfire

from app.agents.state import AgentState
from app.config import settings
from app.gateway import invoke_llm_with_fallback
from app.services.prompts import render_prompt

MAX_EVIDENCE_CHARS = 6000
VERDICTS = ("GROUNDED", "PARTIAL", "UNGROUNDED", "UNKNOWN")

CAUTION_NOTES = {
    "UNGROUNDED": "⚠️ **Fact check failed:** most of this answer could not be verified against the sources; treat it with caution.",
    "PARTIAL": "⚠️ **Caution:** part of this answer could not be verified against the sources.",
}


def _build_evidence(documents: list[dict]) -> str:
    """Numbered evidence block (citations match the responder's numbering)."""
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


def fact_check_node(state: AgentState):
    """Ground the answer against the retrieved chunks and flag issues."""
    if not settings.FACT_CHECK_ENABLED:
        return {"fact_check": "SKIPPED"}

    answer = state.get("final_answer") or ""
    documents = state.get("documents") or []
    if not answer or not documents:
        # Conversational path or no evidence: nothing to verify.
        return {"fact_check": "SKIPPED"}

    evidence = _build_evidence(documents)
    prompt = render_prompt("fact_checker", answer=answer, evidence=evidence)

    with logfire.span("🧪 Fact Check"):
        try:
            verdict = invoke_llm_with_fallback(
                prompt,
                feature="fact_checker",
                slug=state.get("slug"),
                model=state.get("model"),
            ).strip().upper()
        except Exception as e:
            logfire.warning(f"Fact check unavailable: {e}")
            return {
                "fact_check": "UNKNOWN",
                "plan": state["plan"] + ["Fact check: skipped (LLM error)"],
            }

        verdict = verdict.split("\n")[0].strip().split(" ")[0].upper() if verdict else "UNKNOWN"
        if verdict not in VERDICTS:
            verdict = "UNKNOWN"
        logfire.info(f"Fact check verdict: {verdict}")

    plan_update = state["plan"] + [f"Fact check: {verdict}"]
    note = CAUTION_NOTES.get(verdict)
    final_answer = answer
    if note:
        final_answer = answer.rstrip() + "\n\n" + note
        logfire.warning(f"🧪 {note}")

    return {
        "fact_check": verdict,
        "plan": plan_update,
        "final_answer": final_answer,
    }