"""
Retrieval service for the enterprise RAG pipeline.

Replaces the previous `qdrant_service.py` — results now come from Chroma Cloud
via hybrid (BM25 + vector) retrieval.
"""
import logfire

from app.services.retrieval.hybrid_service import hybrid_search


def search_enterprise_knowledge(query: str, limit: int = 15) -> list[dict]:
    """
    Return the most relevant chunks for `query`.

    Uses hybrid retrieval (BM25 + Chroma vector search fused with RRF).

    Returns a list of dicts: {content, source, source_type}.
    """
    with logfire.span("🔎 Search Enterprise Knowledge (Chroma)", query=query, limit=limit):
        results = hybrid_search(query, limit=limit)
        logfire.info(f"Retrieved {len(results)} candidates from Chroma (hybrid).")
        return results