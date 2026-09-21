import logfire

from app.agents.state import AgentState
from app.services.retrieval.chroma_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents_with_sources


def retrieve_node(state: AgentState):
    """
    Performs hybrid retrieval (BM25 + vector) and cross-encoder reranking for
    technical queries.

    Returns structured documents: [{content, source, source_type}] so the
    responder can enforce inline citations.
    """
    query = state["current_query"]

    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Chroma (hybrid) for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Chroma (hybrid)")

        with logfire.span("⚖️ Cross-Encoder Reranking"):
            reranked_docs = rerank_documents_with_sources(query, raw_results, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")

    return {
        "documents": reranked_docs,
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved (BM25 + Vector → Cross-Encoder)"],
    }