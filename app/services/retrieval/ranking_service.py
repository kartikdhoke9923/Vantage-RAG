import time

import logfire
import requests
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.config import settings

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_JINA_RERANK_MODEL = "jina-reranker-v3"
_LOCAL_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_ranker = None
_local_cross_encoder = None


class _JinaReranker:
    """Thin wrapper around the Jina cross-encoder Reranker API."""

    def rerank_indices(self, query: str, documents: list[str], top_n: int) -> list[int]:
        """Score and reorder documents against the query via the Jina API.

        Returns the indices (into `documents`) ranked by relevance, best first.
        """
        response = requests.post(
            _JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _JINA_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        # Each result carries its original `index`; results are already sorted
        # by relevance_score descending.
        return [int(res.get("index")) for res in results[:top_n] if "index" in res]


def _get_ranker() -> _JinaReranker:
    """Returns the Jina Reranker wrapper (lazy singleton)."""
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing Jina Reranker v3 via API...")
        _ranker = _JinaReranker()
    return _ranker


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _rerank_indices(query: str, documents: list[str], top_n: int) -> list[int]:
    """Core Jina API reranking with retry on transient failures."""
    ranker = _get_ranker()
    return ranker.rerank_indices(query, documents, top_n)


def _get_local_cross_encoder():
    """Load the local sentence-transformers cross-encoder (lazy, once)."""
    global _local_cross_encoder
    if _local_cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logfire.info(f"📦 Loading local cross-encoder: {_LOCAL_CROSS_ENCODER_MODEL}")
        _local_cross_encoder = CrossEncoder(_LOCAL_CROSS_ENCODER_MODEL)
    return _local_cross_encoder


def _local_rerank_indices(query: str, documents: list[str], top_n: int) -> list[int]:
    """Local CPU cross-encoder fallback (used when the Jina API is unavailable)."""
    if not documents:
        return []
    model = _get_local_cross_encoder()
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    logfire.info("🧠 Local cross-encoder fallback reranked candidates.")
    return ranked[:top_n]


def rerank_documents_with_sources(query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
    """
    Refines hybrid retrieval results by re-scoring chunks against the query with
    a cross-encoder reranker (Jina API, with a local model fallback).

    `documents` are dicts like {content, source, source_type}. The cross-encoder
    operates on the raw content; the original metadata is re-attached afterwards.

    Falls back to the original retrieval order if reranking fails entirely, so
    the user still receives an answer.
    """
    if not documents:
        return []

    contents = [d["content"] for d in documents]

    # If no Jina key is configured, use the local cross-encoder directly.
    if not settings.JINA_API_KEY:
        logfire.warning("⚠️ JINA_API_KEY not set — using local cross-encoder reranker.")
        indices = _local_rerank_indices(query, contents, top_n)
        return [documents[i] for i in indices]

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to Jina Reranker API...")

    try:
        indices = _rerank_indices(query, contents, top_n)
        duration = time.time() - start_time
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s.")
        return [documents[i] for i in indices]
    except Exception as e:
        logfire.error(f"❌ [Reranker] Jina API failed after retries: {e}. Using local fallback.")
        try:
            indices = _local_rerank_indices(query, contents, top_n)
            return [documents[i] for i in indices]
        except Exception as local_err:
            logfire.error(f"❌ [Reranker] Local fallback also failed: {local_err}")
            # Fallback to the original retrieval order to ensure an answer.
            return documents[:top_n]


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """Compatibility wrapper returning just the reranked contents."""
    dicts = [{"content": doc, "source": "", "source_type": ""} for doc in documents]
    reranked = rerank_documents_with_sources(query, dicts, top_n)
    return [d["content"] for d in reranked]