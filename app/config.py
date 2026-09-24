import json
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Load environment variables from .env file
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = "gpt-4o"  # Default model for GROQ

    # --- Vector DB (Chroma Cloud) ---
    # Replaces the old Qdrant config (QDRANT_URL / QDRANT_API_KEY).
    CHROMA_HOST = os.getenv("CHROMA_HOST", "api.trychroma.com")
    CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
    CHROMA_TENANT = os.getenv("CHROMA_TENANT")
    CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "enterprise_rag")

    # --- Versioned prompts ---
    PROMPTS_DIR = os.getenv("PROMPTS_DIR", "config/prompts")
    # Overrides the active version/registry for prompt selection at runtime.
    PROMPT_VERSION = os.getenv("PROMPT_VERSION")

    # --- LLM Gateway (Portkey) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_PRIMARY_CONFIG_ID = os.getenv("PORTKEY_PRIMARY_CONFIG_ID")
    PORTKEY_PRIMARY_SLUG = os.getenv("PORTKEY_PRIMARY_SLUG", "groq")
    PORTKEY_PRIMARY_MODEL = os.getenv("PORTKEY_PRIMARY_MODEL", "openai/gpt-oss-20b")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # --- Per-role model pool (full @slug/model routing strings) ---
    # One fixed model per role; on failure the gateway rotates to the next model
    # in the pool that isn't in its cool-down window. Empty → default primary.
    PORTKEY_MODEL_GUARDRAIL = os.getenv("PORTKEY_MODEL_GUARDRAIL", "")
    PORTKEY_MODEL_PLANNER = os.getenv("PORTKEY_MODEL_PLANNER", "")
    PORTKEY_MODEL_RESPONDER = os.getenv("PORTKEY_MODEL_RESPONDER", "")

    # --- Eval judge provider (RAGAS Phase 2) ---
    # portkey (prod gateway, shared Groq quota) | groq (direct to api.groq.com,
    # GROQ_FALLBACK key, may share the same org cap) | gemini (OpenAI-compatible
    # endpoint, separate free quota). Decouples eval scoring from the app's own
    # LLM budget so one phase can't starve another.
    EVAL_JUDGE_PROVIDER = os.getenv("EVAL_JUDGE_PROVIDER", "portkey")
    # Optional explicit judge model; provider-specific defaults apply when empty
    # (gemini → "gemini-3.6-flash", groq → PORTKEY_PRIMARY_MODEL).
    EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "")
    # Embeddings backend for RAGAS scoring: "jina" (API, no torch — default) or
    # "hf" (local all-MiniLM via sentence-transformers; needs a working torch).
    EVAL_EMBEDDINGS = os.getenv("EVAL_EMBEDDINGS", "jina")

    # --- Guardrails (NeMo) ---
    # LLM used by the NeMo gate. Empty → provider default per rails.py:
    #   OPENAI_API_KEY set → gpt-5-mini; else GROQ_API_KEY → openai/gpt-oss-20b.
    # Must be a model the provider actually offers (Groq's free tier catalog
    # does not include the old llama-3.1-8b-instant).
    GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL")
    # If the rail provider errors at query time: true → log + proceed to RAG
    # (dev-friendly); false → fail-closed (return 400, deny the request).
    GUARDRAILS_FAIL_OPEN = os.getenv("GUARDRAILS_FAIL_OPEN", "true").lower() == "true"
    # Master switch for the guardrails gate. Set false to skip the gate's LLM
    # call entirely (instant pass-through) — handy on a slow/free provider.
    GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"

    # --- Safety / control layer ---
    # Mask emails, phone numbers and secret-looking tokens in the final answer
    # BEFORE it is returned. Off by default — the résumé contact info is a
    # deliberate part of the docs. Enable when the corpus contains third-party
    # PII or secrets.
    MASK_PII_IN_OUTPUT = os.getenv("MASK_PII_IN_OUTPUT", "false").lower() == "true"
    # Ground the answer's claims against the retrieved chunks with a cheap LLM
    # verdict (GROUNDED / PARTIAL / UNGROUNDED / UNKNOWN) after synthesis.
    FACT_CHECK_ENABLED = os.getenv("FACT_CHECK_ENABLED", "true").lower() == "true"

    # --- LLM gateway hardening ---
    # Redis prompt/response cache for repeated gateway calls (requires Redis).
    GATEWAY_CACHE_ENABLED = os.getenv("GATEWAY_CACHE_ENABLED", "true").lower() == "true"
    GATEWAY_CACHE_TTL = int(os.getenv("GATEWAY_CACHE_TTL", "1800"))

    # --- API auth & production hardening ---
    API_KEY = os.getenv("API_KEY", "")  # Empty = open /query (dev mode)
    STRICT_STARTUP = os.getenv("STRICT_STARTUP", "false").lower() == "true"
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
    # Disable the per-IP slowapi limiter on local/dev boxes — the in-memory
    # storage never clears its window, so sustained eval runs get stuck at 429.
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    POSTGRES_URI = os.getenv("POSTGRES_URI", "")

    @property
    def redis_url(self) -> str:
        return self.REDIS_URL

    @property
    def postgres_uri(self) -> str:
        return self.POSTGRES_URI

    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    LOGFIRE_BASE_URL = os.getenv("LOGFIRE_BASE_URL")

    # --- Deployment / widget / admin ---
    # CORS allow-list (comma separated) for the widget + admin cross-origin calls.
    ALLOWED_ORIGINS = os.getenv(
        "ALLOWED_ORIGINS", "https://kartikworks.co.in,https://www.kartikworks.co.in"
    )
    # Public base URL of the backend (used by widget/admin as the API root).
    APP_BASE_URL = os.getenv("APP_BASE_URL", "")
    # Shared secret between the site admin and /admin-api (the Supabase login is
    # the front door; this is a second capability layer).
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
    # Records chat queries into the chat_logs table (toggle off to save Neon IO).
    ADMIN_LOGGING_ENABLED = os.getenv("ADMIN_LOGGING_ENABLED", "true").lower() == "true"
    # Render injects PORT for the web service; local default is 8000.
    PORT = int(os.getenv("PORT", "8000"))
    # Folder the daily ingestion scheduler re-reads (baked into the image at deploy).
    DATA_DIR = os.getenv("DATA_DIR", "data")
    # Hour (UTC 0-23) the daily re-ingestion tick fires.
    DAILY_INGEST_HOUR = int(os.getenv("DAILY_INGEST_HOUR", "5"))
    # Source type tag applied to ingested docs when data/ has no sub-folders.
    DATA_SOURCE_TYPE = os.getenv("DATA_SOURCE_TYPE", "true")
    # Optional web/structured ingestion sources (the "about me" live data).
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_TABLES = os.getenv("SUPABASE_TABLES", "projects,experience_entries,portfolio_tools,resume")
    WEB_SOURCES_JSON = os.getenv("WEB_SOURCES_JSON", "")

    @property
    def supabase_tables(self) -> list[str]:
        return [t.strip() for t in self.SUPABASE_TABLES.split(",") if t.strip()]

    @property
    def web_sources(self) -> list[dict]:
        raw = (self.WEB_SOURCES_JSON or "").strip()
        if not raw:
            return []
        try:
            items = json.loads(raw)
            return items if isinstance(items, list) else []
        except Exception:  # noqa: BLE001 - bad config should not crash the app
            return []


settings = Settings()  # did becuaue directly from environment variables,
#but now we are using a Settings class to encapsulate the configuration.

# here config files is helping us to get env related variable and others also

# now loading , chunking and services is remaining