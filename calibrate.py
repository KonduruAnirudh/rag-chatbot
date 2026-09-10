# calibrate.py — scratch script for threshold calibration
import asyncio
from app.rag.embed import embed_query
from app.rag.store import store

# Questions you KNOW your documents answer
ANSWERABLE = [
    "What is the shared responsibility model?",
    "How should I manage IAM permissions?",
    "What does the security pillar recommend for incident response?",
]

# Questions clearly outside your corpus
UNANSWERABLE = [
    "What is the recipe for chicken biryani?",
    "Who won the 2019 cricket world cup?",
    "How do I change a car tyre?",
]


async def scores_for(question):
    vec = await embed_query(question)
    results = store.search(vec, top_k=4)
    return [r["score"] for r in results]


async def main():
    print(f"Corpus: {store.count()} chunks\n")

    print("ANSWERABLE (want high scores)")
    best_answerable = []
    for q in ANSWERABLE:
        s = await scores_for(q)
        best_answerable.append(s[0])
        print(f"  {s[0]:.3f} (top)  {[round(x, 3) for x in s]}  {q[:50]}")

    print("\nUNANSWERABLE (want low scores)")
    worst_unanswerable = []
    for q in UNANSWERABLE:
        s = await scores_for(q)
        worst_unanswerable.append(s[0])
        print(f"  {s[0]:.3f} (top)  {[round(x, 3) for x in s]}  {q[:50]}")

    lo = min(best_answerable)
    hi = max(worst_unanswerable)
    print(f"\nWorst answerable top-score:   {lo:.3f}")
    print(f"Best unanswerable top-score:  {hi:.3f}")
    if lo > hi:
        print(f"Clean separation. Set threshold around {(lo + hi) / 2:.2f}")
    else:
        print("OVERLAP — no threshold separates these cleanly.")


asyncio.run(main())