# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Secrets and model selection (from .env) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.6-luna")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- RAG pipeline settings ---
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between adjacent chunks
TOP_K = 4              # chunks retrieved per question
SIMILARITY_THRESHOLD = 0.20  # calibrated against the corpus, see README

def require_api_key() -> str:
    """
    Returns the API key, or raises if it isn't configured.
    Called at the point of use so the app can start without a key.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OPENAI_API_KEY