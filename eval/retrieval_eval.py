# eval/retrieval_eval.py
"""
Retrieval evaluation: dense vs sparse (BM25) vs hybrid.

For each question, find the rank of the first chunk containing the expected
answer. Lower is better; rank <= TOP_K means the answer reaches the model.

Cases are tagged by type:
  keyword    — the question shares vocabulary with the answer (favours BM25)
  paraphrase — the question avoids the document's own terms (favours embeddings)

Run from the project root:
    python -m eval.retrieval_eval
"""
import asyncio
import re

from app.config import TOP_K
from app.rag.embed import embed_query
from app.rag.retrieve import dense_ranking, sparse_ranking, reciprocal_rank_fusion
from app.rag.store import store

# (type, question, regex that only matches a chunk containing the answer)
CASES = [
    # Keyword cases: the question uses the document's own terms
    ("keyword", "What HTTP status code does a recipient return when it accepts a Security Event Token?", r"\b202\b"),
    ("keyword", "What media type is used when transmitting a SET?", r"secevent\+jwt"),
    ("keyword", "Which error code means the SET issuer is not recognised?", r"invalid_issuer"),
    ("keyword", "What does the whitepaper recommend for DDoS incident response?", r"NIST"),
    ("keyword", "Which best practices apply to application layer defense?", r"Application layer defense"),
    ("keyword", "What are AWS WAF rate-based rules used for?", r"rate-based"),

    # Paraphrase cases: the question deliberately avoids the document's terms
    ("paraphrase", "What should a receiver reply if it cannot verify the token's signature?", r"invalid_key"),
    ("paraphrase", "Who at AWS can I contact for help during an ongoing attack?", r"Shield Response Team"),
    ("paraphrase", "How do I stop one client from sending too many requests?", r"rate-based"),
]

METHODS = ["dense", "sparse", "hybrid"]
GROUPS = ["keyword", "paraphrase"]


async def rankings_for(question: str) -> dict[str, list[int]]:
    cosine = store.dense_scores(await embed_query(question))
    dense = dense_ranking(cosine)
    sparse = sparse_ranking(store.sparse_scores(question))
    return {
        "dense": dense,
        "sparse": sparse,
        "hybrid": reciprocal_rank_fusion([dense, sparse]),
    }


def first_match_rank(ranking: list[int], pattern: re.Pattern) -> int | None:
    for rank, index in enumerate(ranking, start=1):
        if pattern.search(store.metadata[index]["text"]):
            return rank
    return None   # never found — e.g. BM25 had no keyword overlap


def summarise(ranks: list[int | None]) -> str:
    recall = sum(1 for r in ranks if r and r <= TOP_K) / len(ranks)
    mrr = sum(1 / r for r in ranks if r) / len(ranks)
    return f"Recall@{TOP_K} = {recall:>4.0%}   MRR = {mrr:.3f}"


async def main():
    if store.is_empty():
        print("Index is empty. Upload documents first.")
        return

    # results[group][method] -> list of ranks
    results = {g: {m: [] for m in METHODS} for g in GROUPS}

    print(f"{store.count()} chunks indexed, TOP_K = {TOP_K}\n")
    header = f"{'type':<11}{'question':<62}" + "".join(f"{m:>9}" for m in METHODS)
    print(header)
    print("-" * len(header))

    for kind, question, expected in CASES:
        pattern = re.compile(expected)

        # Guard against a broken test: the answer must exist in the corpus.
        if not any(pattern.search(m["text"]) for m in store.metadata):
            print(f"{kind:<11}{question[:60]:<62}  SKIPPED - '{expected}' not in any chunk")
            continue

        rankings = await rankings_for(question)
        row = f"{kind:<11}{question[:60]:<62}"
        for method in METHODS:
            rank = first_match_rank(rankings[method], pattern)
            results[kind][method].append(rank)
            row += f"{rank if rank else '-':>9}"
        print(row)

    print("\n'-' means the method never found the answer (counts as a miss).\n")

    for group in GROUPS + ["all"]:
        if group == "all":
            per_method = {m: results["keyword"][m] + results["paraphrase"][m] for m in METHODS}
        else:
            per_method = results[group]

        if not per_method["dense"]:
            continue

        count = len(per_method["dense"])
        print(f"{group.upper()} ({count} questions)")
        for method in METHODS:
            print(f"  {method:<8} {summarise(per_method[method])}")
        print()


if __name__ == "__main__":
    asyncio.run(main())