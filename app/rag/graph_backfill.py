# app/rag/graph_backfill.py
"""
Extract graph records for every chunk already in the vector store.

The chunks indexed before the graph existed never passed through an upload
hook, so they are extracted here in one pass, with the frozen prompt v6.

Output goes to a STAGING file, data/graph_staging/, not to data/graph/: the
records are reviewed before anything becomes part of the graph.

Resumable: records are saved after every batch, and a rerun skips chunks
already extracted with the same prompt version. A failure part-way costs only
the unfinished batch, and chunks that failed are retried by running it again.

Run from the project root:
    python -m app.rag.graph_backfill
"""
import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.config import CHAT_MODEL, EXTRACT_MIN_AGREE, EXTRACT_RUNS, GRAPH_PROMPT_VERSION
from app.rag.graph import chunk_id, record_for
from app.rag.graph_extract import GENERIC_WORDS, extract_many, match_form
from app.rag.store import store

STAGING_DIR = Path("data/graph_staging")
STAGING_PATH = STAGING_DIR / f"extractions-{GRAPH_PROMPT_VERSION}.json"
BATCH_SIZE = 20

# gpt-5.6-luna list prices, USD per million tokens (third-party listings, Sept 2026)
PRICE_INPUT, PRICE_CACHED, PRICE_OUTPUT = 0.20, 0.02, 1.20


def load_staging() -> dict:
    if STAGING_PATH.exists():
        return json.loads(STAGING_PATH.read_text())
    return {"prompt_version": GRAPH_PROMPT_VERSION, "model": CHAT_MODEL,
            "runs": EXTRACT_RUNS, "min_agree": EXTRACT_MIN_AGREE, "chunks": {}}


def save_staging(staging: dict) -> None:
    """Write to a temporary file, then rename: a crash mid-write never leaves half a file."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staging["saved_at"] = datetime.now(timezone.utc).isoformat()
    temporary = STAGING_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(staging))
    temporary.replace(STAGING_PATH)


def summarise(staging: dict, seconds: float) -> None:
    records = staging["chunks"].values()
    failed = [cid for cid, r in staging["chunks"].items() if "error" in r]
    ok = [r for r in records if "error" not in r]
    edges = [e for r in ok for e in r["relations"]]
    verbs = Counter(e["relation"] for e in edges)
    rejected = Counter(d["reason"] for r in ok for d in r["dropped"] if d["kind"] == "relation")
    is_generic = lambda name: match_form(name) in GENERIC_WORDS or match_form(name).removesuffix("s") in GENERIC_WORDS
    generic = sum(1 for e in edges if is_generic(e["subject"]) or is_generic(e["object"]))
    tokens = Counter()
    for r in ok:
        tokens.update(r["usage"])
    cost = ((tokens["input"] - tokens["cached"]) * PRICE_INPUT + tokens["cached"] * PRICE_CACHED
            + tokens["output"] * PRICE_OUTPUT) / 1e6

    print("=" * 78)
    print(f"STAGED: {STAGING_PATH}   (prompt {staging['prompt_version']}, "
          f"{staging['min_agree']} of {staging['runs']} runs)")
    print("-" * 78)
    print(f"  chunks extracted              {len(ok)} of {len(store.metadata)}   failed: {failed or 'none'}")
    print(f"  entities                      {sum(len(r['entities']) for r in ok)}")
    print(f"  edges                         {len(edges)}   (all {staging['runs']} runs: "
          f"{sum(e['votes'] == staging['runs'] for e in edges)}, "
          f"{staging['min_agree']} of {staging['runs']}: {sum(e['votes'] < staging['runs'] for e in edges)})")
    print(f"  chunks with no edges          {sum(1 for r in ok if not r['relations'])}")
    print(f"  verbs                         {dict(verbs.most_common())}")
    print(f"  related_to share              {verbs['related_to']}/{len(edges)} = "
          f"{verbs['related_to'] / len(edges):.0%}" if edges else "  related_to share  n/a")
    print(f"  edges with a generic endpoint {generic}   (admission rule R2 removes these)")
    print(f"  rejections, all runs          {dict(rejected.most_common())}")
    print(f"  tokens                        input {tokens['input']:,} (cached {tokens['cached']:,}), "
          f"output {tokens['output']:,}")
    print(f"  cost                          ${cost:.2f}")
    print(f"  time this session             {seconds / 60:.1f} min")


async def main():
    staging = load_staging()
    if staging["prompt_version"] != GRAPH_PROMPT_VERSION:
        raise SystemExit(f"{STAGING_PATH} holds prompt {staging['prompt_version']}, not {GRAPH_PROMPT_VERSION}.")

    done = {cid for cid, record in staging["chunks"].items() if "error" not in record}
    todo = [c for c in store.metadata if chunk_id(c) not in done]
    print(f"{len(store.metadata)} chunks in the index, {len(done)} already extracted, {len(todo)} to do")

    started = time.time()
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        results = await extract_many(batch)
        for chunk, result in zip(batch, results):
            staging["chunks"][chunk_id(chunk)] = record_for(chunk, result)
        save_staging(staging)
        print(f"  {start + len(batch)}/{len(todo)} chunks   {time.time() - started:.0f}s", flush=True)

    summarise(staging, time.time() - started)


if __name__ == "__main__":
    asyncio.run(main())
