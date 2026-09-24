# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Secrets and model selection (from .env) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.6-luna")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- RAG pipeline settings ---
CHUNK_SIZE = 1000             # characters per chunk
CHUNK_OVERLAP = 150           # characters shared between adjacent chunks
TOP_K = 5                     # chunks retrieved per question
SIMILARITY_THRESHOLD = 0.20   # calibrated: off-topic questions top out around 0.14
MAX_CONTEXT_CHARS = 8000      # hard ceiling on context sent to the LLM

# --- Retrieval mode ---
RETRIEVAL_MODE = "hybrid"   # "dense" (embeddings only) or "hybrid" (embeddings + BM25)
RRF_K = 60                  # Reciprocal Rank Fusion constant; 60 is the standard default

# --- Generation settings ---
TEMPERATURE = 0.2             # answers: faithful and consistent, not robotically rigid
REWRITE_TEMPERATURE = 0.0     # query rewriting: a mechanical transformation, so deterministic
REASONING_EFFORT = "none"     # GPT-5.x rejects `temperature` unless reasoning is off

# --- OCR settings ---
OCR_ENABLED = True
OCR_ENGINE = "vision"    # "vision" (layout-aware, costs tokens) or "tesseract" (local, free)
OCR_MIN_CHARS = 100      # a page with less text than this has no usable text layer
OCR_MAX_PAGES = 20       # bound the worst case on a large scanned document
OCR_CONCURRENCY = 4      # pages OCR'd in parallel; keep low to respect rate limits
VISION_DPI = 150         # vision models read this fine; higher just costs more tokens
TESSERACT_DPI = 300      # classic OCR needs the detail — measured at 88-96% recovery

# --- CORS ---
# Comma-separated origins allowed to call the API from another site.
# Empty (the default) means same-origin only — correct when FastAPI serves the frontend.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]


def require_api_key() -> str:
    """
    Returns the API key, or raises if it isn't configured.
    Called at the point of use so the app can start without a key.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OPENAI_API_KEY