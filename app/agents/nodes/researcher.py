"""Researcher sub-agent — multi-query retrieval.

Decomposes the technical question into up to 3 sub-queries (one cheap LLM
call), runs hybrid retrieval per sub-query through the tool registry, then
dedupes and merges the best chunks into `documents`.
"""
import logfire

from app.agents.state import AgentState
from app.gateway import invoke_llm_with_fallback
from app.services.prompts import render_prompt
from app.tools.registry import retrieve_documents

MAX_SUB_QUERIES = 3
MAX_MERGED = 6


def _decompose(current_query: str) -> list[str]:
    try:
        raw = invoke_llm_with_fallback(
            render_prompt("researcher", question=current_query),
            feature="researcher",
        )
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()][:MAX_SUB_QUERIES]
        return lines or [current_query]
    except Exception as e:  # noqa: BLE001 - researcher must never kill the pipeline
        logfire.warning(f"Sub-query decomposition failed ({e}); using the raw query.")
        return [current_query]


def _merge(docs_by_query: list[list[dict]]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for docs in docs_by_query:
        for doc in docs:
            marker = (doc.get("source", ""), doc.get("content", "")[:80])
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(doc)
        if len(merged) >= MAX_MERGED:
            break
    return merged[:MAX_MERGED]


def researcher_node(state: AgentState):
    """Run sub-query retrieval and merge the best chunks."""
    query = state["current_query"]
    sub_queries = _decompose(query)
    logfire.info(f"🔬 Researcher sub-queries: {sub_queries}")

    docs_by_query = [retrieve_documents(sq, top_n=5) for sq in sub_queries]
    merged = _merge(docs_by_query)
    if not merged:
        merged = retrieve_documents(query, top_n=5)

    logfire.info(f"🔬 Researcher merged {len(merged)} chunks.")
    return {
        "documents": merged,
        "sub_queries": sub_queries,
        "plan": state["plan"] + ["Sub-agent: researcher", f"Sub-queries: {' / '.join(sub_queries)}"],
        "status": f"Researcher gathered {len(merged)} evidence chunks.",
    }