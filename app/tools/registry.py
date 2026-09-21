"""Tool registry for the agent orchestration layer.

Tools are plain callables registered by name. The tool_executor node picks a
tool from the registry based on the orchestrator's intent and stores its output
in the state (documents for retrieval, tool_results for informational tools).
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import logfire

from app.services.retrieval.chroma_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents_with_sources


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]


def retrieve_documents(query: str, top_n: int = 5) -> list[dict]:
    """Hybrid retrieval (BM25 + vector → cross-encoder rerank).

    Returns [{content, source, source_type}] for the responder's numbered
    citation context.
    """
    with logfire.span("🧰 Tool: retrieve_documents", query=query, top_n=top_n):
        raw = search_enterprise_knowledge(query, limit=15)
        reranked = rerank_documents_with_sources(query, raw, top_n=top_n)
        logfire.info(f"Tool 'retrieve_documents' → {len(reranked)} chunks.")
        return reranked


def list_sources(top_n: int = 40) -> str:
    """List the unique documents/chunks currently indexed in Chroma.

    Returns a human-readable inventory for the responder to summarize.
    """
    from app.services.retrieval.chroma_client import get_or_create_collection

    collection = get_or_create_collection()
    counts: dict[str, int] = {}
    offset = 0
    page_size = 300
    while True:
        page = collection.get(include=["metadatas"], limit=page_size, offset=offset)
        metas = page.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            source = m.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        if len(metas) < page_size:
            break
        offset += page_size

    rows = [f"- {name} ({count} chunks)" for name, count in sorted(counts.items())]
    header = f"Available documents ({len(counts)} sources):" if rows else "No documents indexed yet."
    return header + ("\n" + "\n".join(rows) if rows else "")


TOOL_REGISTRY: dict[str, Tool] = {
    "retrieve_documents": Tool(
        name="retrieve_documents",
        description="Hybrid retrieval across the documentation corpus; returns the most relevant chunks for a query.",
        handler=retrieve_documents,
    ),
    "list_sources": Tool(
        name="list_sources",
        description="Lists every document/chunk currently indexed in the knowledge base.",
        handler=list_sources,
    ),
}