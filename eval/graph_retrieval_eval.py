# eval/graph_retrieval_eval.py
"""
Step 7: does graph retrieval promote the evidence today's retrieval ranks too low?

Retrieval only - no answers are generated. Every arm starts from today's retrieval
(A0), unchanged. A graph arm ADDS at most 2 chunks after A0's top 5, so its whole
contribution is those 2 chunks: they can be attributed, and removed.

  A0         today: dense + BM25 + RRF, cosine gate, top 5 - the production code, called as is
  A0@7       the same with the top 7: does showing the model two more chunks do as well?
  G1 ...     A0's top 5, then up to 2 graph chunks, on every question         (A2)
  G1* ...    the same, only on questions labelled needs_graph                  (the ceiling)

The graph side, as pre-registered in the README ("Step 7 design"):
  1. link   the entities named in the question (eval/linking_recall.link, unchanged)
  2. reach  the chunks whose quote supports an edge reached from them
              G1    1 hop
              G2-U  2 hops
              G2-H  2 hops, never through a hub (10 or more neighbours)
              G2-W  2 hops, each step weighted 1 / neighbour count
  3. gate   a graph chunk must pass the production cosine gate and not be in A0's top 5
  4. rank   reached from the most entities first, then the configuration's score, then cosine
  5. add    the top 2, after A0's top 5
Relation verbs are never used: only which chunk quotes a link between which two entities.

A configuration is CHOSEN on the development half only. The held-out half evaluates
only the configuration chosen there (PRESELECTED) and never selects one: choosing
with held-out data would turn the test set into a tuning set.

Every run is saved to eval/graph_runs/, with the quotes behind each added chunk.

Run from the project root:
    python -m eval.graph_retrieval_eval                   # Phase 1: the development half
    python -m eval.graph_retrieval_eval --heldout-once    # Phase 3 only: the held-out half, run once
"""
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from app.config import SIMILARITY_THRESHOLD, TOP_K
from app.rag import retrieve as R
from app.rag.graph import graph_store
from eval.check_graph_questions import DOCUMENTS
from eval.linking_recall import classify, entity_forms, link

QUESTIONS = Path("eval/graph_questions.yaml")
RUNS = Path("eval/graph_runs")
ADD = 2                         # graph chunks added after A0's top 5
CONTROL_K = TOP_K + ADD         # A0@7: the same number of chunks, no graph
HUB = 10                        # neighbours; G2-H never traverses through such a node
LATENCY_LIMIT_MS = 500
NOISE_MARGIN = 0.10
CONFIGS = ("G1", "G2-U", "G2-H", "G2-W")      # also the simplicity order that breaks ties
# Chosen on the development half by the pre-registered rule (Phase 1,
# eval/graph_runs/dev-20260925-230614.json). The held-out half evaluates only this.
PRESELECTED = "G1"
EXISTING = {f"Q{i:02d}" for i in range(1, 10)}


def configs_for(split: str) -> tuple[str, ...]:
    """Every configuration competes on the development half; the held-out half tests only the one chosen there."""
    return CONFIGS if split == "dev" else (PRESELECTED,)


# --- the graph side ------------------------------------------------------------

def pair_chunks(graph, index_of) -> dict:
    """(entity, neighbour) -> store indices of the chunks quoting an edge between them."""
    chunks = defaultdict(set)
    for (subject, _, obj), provenance in graph.edges.items():
        found = {index_of[(p["doc_id"], p["chunk_index"])] for p in provenance}
        chunks[(subject, obj)] |= found
        chunks[(obj, subject)] |= found
    return chunks


def reach(config: str, seeds: list[str], graph, pairs: dict) -> dict:
    """Chunk index -> which starting entities reach it, in how few hops, with what weight, via which links."""
    found = defaultdict(lambda: {"seeds": set(), "hops": 9, "weight": 0.0, "via": set()})

    def add(link_pair, seed, hops, weight):
        for chunk in pairs[link_pair]:
            f = found[chunk]
            f["seeds"].add(seed)
            f["hops"] = min(f["hops"], hops)
            f["weight"] += weight
            f["via"].add(link_pair)

    for seed in seeds:
        first = graph.adjacency.get(seed, set())
        for middle in first:
            step = 1 / len(first)
            add((seed, middle), seed, 1, step)
            if config == "G1" or (config == "G2-H" and graph.neighbour_count(middle) >= HUB):
                continue
            second = graph.adjacency[middle]
            for end in second:
                if end != seed:
                    add((middle, end), seed, 2, step / len(second))
    return found


def rank_key(config: str, found: dict, cosine):
    score = {"G1": lambda f: 0, "G2-U": lambda f: -f["hops"], "G2-H": lambda f: -f["hops"],
             "G2-W": lambda f: f["weight"]}[config]
    return lambda chunk: (len(found[chunk]["seeds"]), score(found[chunk]), float(cosine[chunk]))


def graph_side(config, question, forms, graph, pairs, cosine, top5: set):
    """Steps 1-5 for one question. Returns the added chunks, what was reached, and the time taken."""
    started = time.perf_counter()
    seeds = link(question, forms)
    found = reach(config, seeds, graph, pairs)
    eligible = [c for c in found if cosine[c] >= SIMILARITY_THRESHOLD and c not in top5]
    added = sorted(eligible, key=rank_key(config, found, cosine), reverse=True)[:ADD]
    return added, found, (time.perf_counter() - started) * 1000


# --- running the arms --------------------------------------------------------

def label(chunk: dict) -> str:
    prefix = {filename: p for p, filename in DOCUMENTS.items()}
    return f"{prefix.get(chunk['filename'], chunk['filename'])}:{chunk['chunk_index']}"


def quotes_for(chunk: int, via: set, graph, metadata) -> list[str]:
    """The edge quotes that brought a chunk in (verbs shown for reading only; never used)."""
    doc, number = metadata[chunk]["doc_id"], metadata[chunk]["chunk_index"]
    out = []
    for (subject, verb, obj), provenance in graph.edges.items():
        if (subject, obj) in via or (obj, subject) in via:
            out += [f"{subject} --{verb}--> {obj} | {p['evidence']}" for p in provenance
                    if p["doc_id"] == doc and p["chunk_index"] == number]
    return out


async def run(questions, configs: tuple[str, ...]):
    graph = graph_store.graph
    metadata = R.store.metadata
    index_of = {(c["doc_id"], c["chunk_index"]): i for i, c in enumerate(metadata)}
    gold_index = {label(c): i for i, c in enumerate(metadata)}
    pairs = pair_chunks(graph, index_of)
    forms = entity_forms(graph)
    as_indices = lambda chunks: [index_of[(c["doc_id"], c["chunk_index"])] for c in chunks]

    # Production retrieve() runs unchanged; each question is embedded once, so every arm sees one vector
    vectors, real_embed = {}, R.embed_query

    async def embed_once(text):
        if text not in vectors:
            vectors[text] = await real_embed(text)
        return vectors[text]

    R.embed_query = embed_once
    results = []
    try:
        for q in questions:
            a0 = as_indices(await R.retrieve(q["question"]))
            ranked = as_indices(await R.retrieve(q["question"], top_k=len(metadata)))   # every chunk past the gate
            assert a0 == ranked[:TOP_K], "A0 differs from its own full ranking"
            cosine = R.store.dense_scores(await embed_once(q["question"]))
            linked = link(q["question"], forms)
            gold = [gold_index[g] for g in q["gold_chunks"]]
            record = {
                "id": q["id"], "category": q["category"], "kind": q["off_topic_kind"],
                "needs_graph": q["needs_graph"], "gold": gold,
                "gold_labels": [f"{g} (A0 rank {ranked.index(i) + 1 if i in ranked else 'below gate'})"
                                for g, i in zip(q["gold_chunks"], gold)],
                "linking": [(name, *classify(name, linked, graph)) for name in q["entities"]],
                "seeds": linked,
                "contexts": {"A0": a0, "A0@7": ranked[:CONTROL_K]},
                "added": {}, "latency_ms": {}, "candidates": {},
            }
            for config in configs:
                added, found, ms = graph_side(config, q["question"], forms, graph, pairs, cosine, set(a0))
                if not a0:
                    assert not added, "a graph chunk passed the gate although A0 found none"
                record["contexts"][config] = a0 + added
                record["contexts"][config + "*"] = a0 + added if q["needs_graph"] else a0
                record["latency_ms"][config] = ms
                record["candidates"][config] = len(found)
                record["added"][config] = [{
                    "chunk": label(metadata[c]), "gold": c in gold, "cosine": round(float(cosine[c]), 3),
                    "from": sorted(found[c]["seeds"]), "hops": found[c]["hops"],
                    "quotes": quotes_for(c, found[c]["via"], graph, metadata),
                } for c in added]
            results.append(record)
    finally:
        R.embed_query = real_embed
    return results


# --- scoring -------------------------------------------------------------------

def score(results, arm: str) -> dict:
    config = arm.rstrip("*")
    is_graph = config in CONFIGS
    with_gold = [r for r in results if r["gold"]]
    multi = [r for r in with_gold if r["category"] == "multi-hop"]
    full = lambda r: set(r["gold"]) <= set(r["contexts"][arm])
    found = lambda r: len(set(r["gold"]) & set(r["contexts"][arm]))
    counted = [r for r in with_gold if r["needs_graph"] or not arm.endswith("*")]
    added = [a for r in counted for a in r["added"][config]] if is_graph else []
    latencies = sorted(r["latency_ms"][config] for r in results) if is_graph else []
    return {
        "multi_full": sum(map(full, multi)), "multi_n": len(multi),
        "graph_full": sum(full(r) for r in multi if r["needs_graph"]),
        "graph_n": sum(r["needs_graph"] for r in multi),
        "gold_found": sum(map(found, with_gold)), "gold_n": sum(len(r["gold"]) for r in with_gold),
        "noise": 1 - sum(map(found, with_gold)) / sum(len(r["contexts"][arm]) for r in with_gold),
        "added": len(added), "added_gold": sum(a["gold"] for a in added),
        "top5_same": all(r["contexts"][arm][:TOP_K] == r["contexts"]["A0"] for r in results if r["id"] in EXISTING),
        "refused": all(not r["contexts"][arm] for r in results if r["kind"] == "names-nothing"),
        "entity_added": {r["id"]: len(r["added"][config]) for r in results if r["kind"] == "names-entity"}
                        if is_graph and not arm.endswith("*") else {},
        "p50": statistics.median(latencies) if latencies else None,
        "p95": latencies[max(0, round(0.95 * len(latencies)) - 1)] if latencies else None,
        "candidates": statistics.median(r["candidates"][config] for r in results) if is_graph else None,
    }


def select(scores: dict, split: str):
    """The pre-registered rule, applied mechanically to the A2 arms - on the development half only."""
    if split != "dev":
        raise ValueError(f"a configuration is chosen on the development half only, not on '{split}'")
    notes, passing = [], []
    for config in CONFIGS:
        s = scores[config]
        failures = [why for failed, why in (
            (not s["refused"], "changes refusal of a question naming nothing"),
            (not s["top5_same"], "changes the existing questions' top 5"),
            (s["p95"] >= LATENCY_LIMIT_MS, f"p95 latency {s['p95']:.0f} ms"),
        ) if failed]
        notes.append(f"{config}: " + ("; ".join(failures) if failures else "passes the hard criteria"))
        if not failures:
            passing.append(config)
    if not passing:
        return None, notes
    precision = lambda c: scores[c]["added_gold"] / scores[c]["added"] if scores[c]["added"] else 0.0
    chosen = max(passing, key=lambda c: (scores[c]["multi_full"], precision(c), -CONFIGS.index(c)))
    return chosen, notes


# --- report --------------------------------------------------------------------

def report(results, split: str, configs: tuple[str, ...]):
    arms = ["A0", "A0@7", *configs, *(c + "*" for c in configs)]
    scores = {arm: score(results, arm) for arm in arms}
    print(f"STEP 7 - {split} half, {len(results)} questions, retrieval only\n")

    print("LINKING, by question type (intended entities linked)")
    rows = defaultdict(Counter)
    for r in results:
        outcomes = [outcome for _, outcome, _ in r["linking"]]
        for group in [r["category"]] + (["needs graph"] if r["needs_graph"] else []):
            rows[group]["intended"] += len(outcomes)
            rows[group]["linked"] += outcomes.count("linked")
            rows[group]["questions"] += bool(outcomes)
            rows[group]["all"] += bool(outcomes) and all(o == "linked" for o in outcomes)
    for group in ("plain", "relationship", "multi-hop", "needs graph", "off-topic"):
        c = rows[group]
        if c["intended"]:
            print(f"  {group:<14}{c['linked']:>3}/{c['intended']:<3}{c['linked'] / c['intended']:>5.0%}"
                  f"   every entity linked in {c['all']}/{c['questions']} questions")

    print("\nMULTI-HOP, per question   (Y = every gold chunk is in the context)")
    columns = ["A0", "A0@7", *configs]
    print(f"  {'id':<5}{'graph':<7}{'linked':<8}" + "".join(f"{a:<6}" for a in columns) + "gold")
    for r in results:
        if r["category"] != "multi-hop":
            continue
        linked = f"{sum(o == 'linked' for _, o, _ in r['linking'])}/{len(r['linking'])}"
        cells = "".join(f"{'Y' if set(r['gold']) <= set(r['contexts'][a]) else '.':<6}" for a in columns)
        print(f"  {r['id']:<5}{'yes' if r['needs_graph'] else 'no':<7}{linked:<8}{cells}" + "; ".join(r["gold_labels"]))
        for name, outcome, _ in r["linking"]:
            if outcome != "linked":
                print(f"{'':<20}not linked: {name!r} ({outcome})")

    print("\nWHAT EACH CONFIGURATION ADDED   (+chunk, * = gold)")
    for r in results:
        kind = f"/{r['kind']}" if r["kind"] else ""
        print(f"  {r['id']} [{r['category']}{kind}]  linked: {', '.join(r['seeds']) or 'nothing'}")
        for config in configs:
            added = " ".join(f"+{a['chunk']}{'*' if a['gold'] else ''}" for a in r["added"][config]) or "-"
            print(f"      {config:<5} {added:<22} (candidates {r['candidates'][config]})")

    print("\nSUMMARY   context = what the model would see; noise is pooled over questions with gold chunks")
    print(f"  {'arm':<7}{'multi-hop':>10}{'needs graph':>12}{'gold found':>11}{'noise':>7}{'added gold':>11}"
          f"{'top-5 same':>11}{'refused':>8}{'p50/p95 ms':>12}{'candidates':>11}")
    for arm in arms:
        s = scores[arm]
        added = f"{s['added_gold']}/{s['added']}" if arm not in ("A0", "A0@7") else ""
        latency = f"{s['p50']:.1f}/{s['p95']:.1f}" if s["p50"] is not None else ""
        candidates = f"{s['candidates']:.0f}" if s["candidates"] is not None else ""
        print(f"  {arm:<7}{s['multi_full']:>6}/{s['multi_n']:<3}{s['graph_full']:>8}/{s['graph_n']:<3}"
              f"{s['gold_found']:>7}/{s['gold_n']:<3}{s['noise']:>7.0%}{added:>11}{'yes' if s['top5_same'] else 'NO':>11}"
              f"{'yes' if s['refused'] else 'NO':>8}{latency:>12}{candidates:>11}")
    print("\n  graph chunks added to off-topic questions naming a corpus entity (recorded, no pass/fail):")
    for config in configs:
        print(f"    {config:<5} {scores[config]['entity_added']}")

    a0, a07 = scores["A0"], scores["A0@7"]
    if split == "dev":
        print("\nSELECTION   the pre-registered rule, applied to the A2 arms")
        chosen, notes = select(scores, split)
        for note in notes:
            print(f"  {note}")
    else:
        chosen = PRESELECTED
        print(f"\nTESTED CONFIGURATION   {chosen}, chosen on the development half; "
              "no configuration is selected on held-out data")
    if chosen:
        s = scores[chosen]
        print(f"  -> {'chosen' if split == 'dev' else 'tested'}: {chosen}")
        print(f"  multi-hop fully recalled: {chosen} {s['multi_full']}/{s['multi_n']}, "
              f"A0 {a0['multi_full']}/{a0['multi_n']}, A0@7 {a07['multi_full']}/{a07['multi_n']}"
              + ("   (the +15-point test is on the held-out half)" if split == "dev" else ""))
        if split == "dev":
            noise_ok = s["noise"] <= a0["noise"] + NOISE_MARGIN
            print(f"  noise under A2: {s['noise']:.0%} against A0's {a0['noise']:.0%} + 10 points -> "
                  + ("passes, so A3 is skipped, as pre-registered" if noise_ok
                     else "FAILS, so A3's trigger is designed next, from development results only"))
    else:
        print("  -> no configuration passes the hard criteria")
    return scores, chosen


def main():
    split = "heldout" if "--heldout-once" in sys.argv else "dev"
    configs = configs_for(split)
    questions = [q for q in yaml.safe_load(QUESTIONS.read_text()) if q["keep"] and q["split"] == split]
    results = asyncio.run(run(questions, configs))
    scores, chosen = report(results, split, configs)

    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{split}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"split": split, "configs": list(configs), "chosen": chosen,
                                "scores": scores, "questions": results},
                               indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
