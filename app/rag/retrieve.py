# app/rag/retrieve.py
from collections import defaultdict

import numpy as np

from app.config import TOP_K, SIMILARITY_THRESHOLD, RETRIEVAL_MODE, RRF_K
from app.rag.embed import embed_query
from app.rag.store import store


def dense_ranking(cosine: np.ndarray) -> list[int]:
    """Every chunk index, most semantically similar first."""
    return [int(i) for i in np.argsort(cosine)[::-1]]


def sparse_ranking(bm25: np.ndarray) -> list[int]:
    """Chunk indexes with at least one keyword match, best match first."""
    order = np.argsort(bm25)[::-1]
    return [int(i) for i in order if bm25[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """
    Combine rankings by position, not by score. Cosine (0 to 1) and BM25
    (unbounded) aren't comparable, but ranks always are. A chunk ranked
    well by both lists rises to the top; strong in one still gets in.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] += 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True)


async def retrieve(
    question: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    mode: str = RETRIEVAL_MODE,
) -> list[dict]:
    """
    Find the chunks most relevant to a question.

    Ordering comes from dense search, or from dense + BM25 fused. The
    RELEVANCE GATE is always the calibrated cosine threshold: fused scores
    have no absolute meaning, so they can't decide whether anything is
    relevant at all. An empty list means refuse without calling the LLM.
    """
    if store.is_empty():
        return []

    query_vector = await embed_query(question)
    cosine = store.dense_scores(query_vector)

    if mode == "hybrid":
        order = reciprocal_rank_fusion([
            dense_ranking(cosine),
            sparse_ranking(store.sparse_scores(question)),
        ])
    else:
        order = dense_ranking(cosine)

    results = []
    for index in order:
        if cosine[index] < threshold:
            continue
        results.append({**store.metadata[index], "score": float(cosine[index])})
        if len(results) == top_k:
            break
    return results