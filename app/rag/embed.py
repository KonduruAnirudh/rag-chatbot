# app/rag/embed.py
import numpy as np
from openai import AsyncOpenAI

from app.config import require_api_key, EMBEDDING_MODEL

client = AsyncOpenAI(api_key=require_api_key())

BATCH_SIZE = 64


class EmbeddingError(Exception):
    """Raised when the embedding API call fails."""


async def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts.
    Returns an (n, dim) float32 array of unit-length vectors.
    """
    if not texts:
        raise EmbeddingError("No texts to embed.")

    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
        except Exception as e:
            print(f"[EMBEDDING ERROR] {type(e).__name__}: {e}")
            raise EmbeddingError("Could not generate embeddings.")

        # The API returns results in the same order as the input.
        vectors.extend([item.embedding for item in response.data])

    array = np.array(vectors, dtype=np.float32)
    return _normalize(array)


async def embed_query(text: str) -> np.ndarray:
    """Embed a single query. Returns a 1-D vector of shape (dim,)."""
    result = await embed_texts([text])
    return result[0]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so cosine similarity is a dot product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0          # avoid dividing by zero
    return vectors / norms