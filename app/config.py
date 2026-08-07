import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    # Load environment variables from .env file
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"  # Default collection name

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = "gpt-4o"  # Default model for GROQ

settings = Settings()  # did becuaue directly from environment variables, 
#but now we are using a Settings class to encapsulate the configuration.

# here config files is helping us to get env related variable and others also

# now loading , chunking and services is remaining