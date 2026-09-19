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

    # --- Guardrails (NeMo) ---
    # LLM used by the NeMo gate. Empty → provider default per rails.py:
    #   OPENAI_API_KEY set → gpt-5-mini; else GROQ_API_KEY → openai/gpt-oss-20b.
    # Must be a model the provider actually offers (Groq's free tier catalog
    # does not include the old llama-3.1-8b-instant).
    GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL")
    # If the rail provider errors at query time: true → log + proceed to RAG
    # (dev-friendly); false → fail-closed (return 400, deny the request).
    GUARDRAILS_FAIL_OPEN = os.getenv("GUARDRAILS_FAIL_OPEN", "true").lower() == "true"

    # --- API auth & production hardening ---
    API_KEY = os.getenv("API_KEY", "")  # Empty = open /query (dev mode)
    STRICT_STARTUP = os.getenv("STRICT_STARTUP", "false").lower() == "true"
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
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


settings = Settings()  # did becuaue directly from environment variables,
#but now we are using a Settings class to encapsulate the configuration.

# here config files is helping us to get env related variable and others also

# now loading , chunking and services is remaining