"""
Hybrid retrieval: BM25 lexical + Chroma vector search fused with
Reciprocal Rank Fusion (RRF).

The BM25 corpus is built once per process by pulling every chunk from the
Chroma collection via `get()`, so the lexical index always mirrors whatever
was last ingested.
"""
import re

import logfire
from rank_bm25 import BM25Okapi

from app.services.retrieval.chroma_client import get_or_create_collection
from app.services.retrieval.embedding import embed_query

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_bm25_index: BM25Okapi | None = None
_corpus_ids: list | None = None
_corpus_docs: list | None = None


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization used by the BM25 index."""
    return _TOKEN_RE.findall(text.lower())


def _load_corpus() -> tuple[list, list]:
    """Pull all chunks from Chroma once (paginated); return (ids, docs)."""
    global _corpus_ids, _corpus_docs
    if _corpus_ids is None or _corpus_docs is None:
        collection = get_or_create_collection()
        total = collection.count()
        max_limit = 300  # Chroma Cloud caps a single get() at 300 rows.
        ids: list = []
        documents: list = []
        metadatas: list = []
        for offset in range(0, total, max_limit):
            page = collection.get(
                include=["documents", "metadatas"],
                limit=max_limit,
                offset=offset,
            )
            ids.extend(page["ids"])
            documents.extend(page["documents"] or [])
            metadatas.extend(page["metadatas"] or [])

        docs = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            docs.append(
                {
                    "id": doc_id,
                    "content": documents[i] if i < len(documents) else "",
                    "source": meta.get("source", ""),
                    "source_type": meta.get("source_type", ""),
                }
            )
        _corpus_ids = ids
        _corpus_docs = docs
        logfire.info(f"Hybrid corpus loaded from Chroma: {len(ids)}/{total} chunks.")
    return _corpus_ids, _corpus_docs


def get_bm25_index() -> BM25Okapi:
    """Lazy in-memory BM25 index built from the Chroma corpus."""
    global _bm25_index
    if _bm25_index is None:
        _, docs = _load_corpus()
        tokenized = [tokenize(d["content"]) for d in docs]
        _bm25_index = BM25Okapi(tokenized)
    return _bm25_index


def reset_hybrid_cache():
    """Clear the cached BM25 index/corpus (call after re-indexing)."""
    global _bm25_index, _corpus_ids, _corpus_docs
    _bm25_index = None
    _corpus_ids = None
    _corpus_docs = None


def _rrf_rank(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion contribution for a 0-indexed rank."""
    return 1.0 / (k + rank + 1)


def _vector_candidates(query: str, top_k: int) -> list[dict]:
    """Top-k dense candidates from Chroma with metadata."""
    collection = get_or_create_collection()
    query_embedding = embed_query(query)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]

    candidates = []
    for i, doc_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        candidates.append(
            {
                "id": doc_id,
                "content": documents[i] if i < len(documents) else "",
                "source": meta.get("source", ""),
                "source_type": meta.get("source_type", ""),
            }
        )
    return candidates


def _bm25_candidates(query: str, top_k: int) -> list[dict]:
    """Top-k lexical candidates from the BM25 index (score > 0 only)."""
    bm25 = get_bm25_index()
    ids, docs = _load_corpus()
    if not docs:
        return []

    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked = [i for i in ranked if scores[i] > 0][:top_k]

    return [
        {
            "id": ids[i],
            "content": docs[i]["content"],
            "source": docs[i]["source"],
            "source_type": docs[i]["source_type"],
            "bm25_score": float(scores[i]),
        }
        for i in ranked
    ]


def hybrid_search(
    query: str,
    limit: int = 15,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Fuse dense (Chroma) + sparse (BM25) candidates via Reciprocal Rank Fusion.

    Returns top `limit` results as [{id, content, source, source_type}].
    """
    with logfire.span("🤝 Hybrid Search (BM25 + Vector)", query=query, limit=limit):
        vector_hits = _vector_candidates(query, vector_top_k)
        bm25_hits = _bm25_candidates(query, bm25_top_k)

        fused: dict[str, dict] = {}
        # Dense candidates keep their ordering from the vector store.
        for rank, cand in enumerate(vector_hits):
            entry = fused.setdefault(cand["id"], dict(cand))
            entry["rrf_score"] = entry.get("rrf_score", 0.0) + _rrf_rank(rank, rrf_k)

        # Lexical candidates are ranked by BM25 score descending.
        for rank, cand in enumerate(bm25_hits):
            entry = fused.setdefault(cand["id"], dict(cand))
            entry["rrf_score"] = entry.get("rrf_score", 0.0) + _rrf_rank(rank, rrf_k)

        ranked = sorted(
            fused.values(), key=lambda c: c.get("rrf_score", 0.0), reverse=True
        )[:limit]
        logfire.info(
            f"Hybrid: {len(vector_hits)} dense + {len(bm25_hits)} lexical → {len(ranked)} fused."
        )
        return ranked