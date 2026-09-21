"""
Shared Chroma Cloud client + collection access.

Connection parameters come from settings (CHROMA_HOST / CHROMA_API_KEY /
CHROMA_TENANT / CHROMA_DATABASE) — previously Qdrant, now Chroma Cloud.
"""
import chromadb
import logfire

from app.config import settings

_client = None
_collection = None
_embedding_function = None


def get_chroma_client():
    """Lazy singleton Chroma Cloud client."""
    global _client
    if _client is None:
        _client = chromadb.CloudClient(
            tenant=settings.CHROMA_TENANT,
            database=settings.CHROMA_DATABASE,
            api_key=settings.CHROMA_API_KEY,
            cloud_host=settings.CHROMA_HOST,
            cloud_port=443,
            enable_ssl=True,
        )
        logfire.info(
            "Chroma Cloud client initialized",
            host=settings.CHROMA_HOST,
            database=settings.CHROMA_DATABASE,
            collection=settings.CHROMA_COLLECTION,
        )
    return _client


def get_or_create_collection():
    """
    Get (or lazily create) the enterprise RAG collection.

    embedding_function=None → the client never runs a default local embedding
    model; every add/query must pass explicit embeddings (which this project
    does via jina-embeddings-v3 / mxbai fallback).
    """
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_embedding_function,
        )
        logfire.info(f"Collection '{settings.CHROMA_COLLECTION}' is ready (cosine space).")
    return _collection


def delete_collection():
    """Drop the collection so the next ingest starts from a clean slate."""
    global _collection
    client = get_chroma_client()
    existing_names = {c.name for c in client.list_collections()}
    if settings.CHROMA_COLLECTION in existing_names:
        client.delete_collection(settings.CHROMA_COLLECTION)
        logfire.info(f"Collection '{settings.CHROMA_COLLECTION}' deleted.")
    _collection = None


def reset_chroma_cache():
    """Reset cached client/collection (used by tests or after a wipe)."""
    global _client, _collection
    _client = None
    _collection = None