# app/rag/retrieve.py
from app.config import TOP_K, SIMILARITY_THRESHOLD
from app.rag.embed import embed_query
from app.rag.store import store


async def retrieve(
    question: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Find the chunks most relevant to a question.

    Returns chunks scoring above the threshold, best first.
    An empty list means nothing in the corpus is relevant —
    the caller should refuse rather than ask the LLM to guess.
    """
    if store.is_empty():
        return []

    query_vector = await embed_query(question)
    results = store.search(query_vector, top_k=top_k)

    return [r for r in results if r["score"] >= threshold]