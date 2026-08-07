













def __probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return None

def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")

def _init():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = __probe_gemini()
        if _embedding_model is None:
            _embedding_model = _load_fallback()

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    if _embedding_model is None:
        return 0
    if hasattr(_embedding_model, "embedding_dim"):
        return _embedding_model.embedding_dim
    if hasattr(_embedding_model, "get_sentence_embedding_dimension"):
        return _embedding_model.get_sentence_embedding_dimension()
    return 3072  # default for gemini-embedding-2-preview