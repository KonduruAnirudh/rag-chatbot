# eval/extract_pilot.py
"""
Step 5 pilot: entity and relation extraction on hand-picked chunks.

Prints everything, so extraction quality can be judged before any of it
reaches a graph. It saves only pilot artefacts, never graph data:
  eval/pilot_runs/<version>-<time>.json   this run's edges, ids and votes, so
                                          labels can be scored and the next
                                          run can show which verbs changed
  eval/pilot_<version>_for_labelling.txt  the chunk listing to label, without
                                          vote counts

Chunks come in two groups:
  known     chosen during design for the traps they contain; the prompt and
            checks were tuned on them, so they are not an independent test
  holdout   drawn at random (seeded) after the prompt and checks were fixed;
            precision here shows whether the fixes generalise

Every extraction is itself EXTRACT_RUNS model calls, keeping the edges at
least EXTRACT_MIN_AGREE of them found. The pilot does two such extractions:
  A          is printed in full for labelling
  A vs B     measures whether the VOTED output is stable — what a rebuild of
             the graph would actually see
  inside A   agreement between single runs, the model's raw consistency

Run from the project root:
    python -m eval.extract_pilot
"""
import asyncio
import contextlib
import io
import itertools
import json
import textwrap
import typing
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import EXTRACT_MIN_AGREE, EXTRACT_RUNS, GRAPH_PROMPT_VERSION
from app.rag.graph_extract import RelationType, clean, edge_key, extract_many, match_form
from app.rag.store import store

RUNS_DIR = Path("eval/pilot_runs")
FIXTURES = {
    # scanned5.pdf has no text layer; its OCR output was frozen once, because
    # vision OCR is not deterministic and the holdout must not change per run.
    "fixture:scanned5": Path("eval/pilot_fixtures/scanned5_chunks.json"),
}

# (source, chunk_index, group, the trap or reason this chunk is here)
PILOT = [
    ("AWS-WhitePaper-Sample.pdf", 39,  "known",   "Shield Standard vs Shield Advanced vs AWS WAF; 'mitigate'; running header"),
    ("AWS-WhitePaper-Sample.pdf", 153, "known",   "Security Hub CSPM / Firewall Manager; a negation that must not become an edge"),
    ("AWS-WhitePaper-Sample.pdf", 154, "known",   "finding -> Security Hub CSPM; an explicit 'after' sequence; footer; pronoun"),
    ("AWS-WhitePaper-Sample.pdf", 158, "known",   "SRT escalation (conditional); support-plan noise; footer"),
    ("rfc8935.pdf",               15,  "known",   "different domain: SET Recipient, 400 Bad Request"),
    ("AWS-WhitePaper-Sample.pdf", 156, "known",   "conforms_to NIST; a list of steps that must NOT become precedes"),
    ("AWS-WhitePaper-Sample.pdf", 157, "known",   "an explicit 'before': a candidate for precedes"),
    # Holdout, drawn with random.Random(2026) after prompt v6 was fixed. Pools:
    # chunks of 600+ characters with no URLs, never used or read during design;
    # whitepaper pages 21-25 excluded (scanned5.pdf duplicates them) and 59-61
    # (the known chunks); OCR chunks overlapping a known chunk excluded.
    ("AWS-WhitePaper-Sample.pdf", 23,  "holdout", "random whitepaper chunk"),
    ("rfc8935.pdf",               20,  "holdout", "random RFC chunk"),
    ("fixture:scanned5",          11,  "holdout", "random OCR'd chunk (vision OCR of scanned5.pdf)"),
]
GROUPS = ("known", "holdout")

# Baseline: prompt v5 with the same 2-of-3 vote, on the 7 known chunks
BASE_LABEL, BASE_VOTED, BASE_RAW = "v5, known chunks", "0.57", "0.44"

# Targets agreed in the Step 5 design
TARGET_QUOTED = 0.95
TARGET_RELATED_TO = 0.15
TARGET_JACCARD = 0.80

# gpt-5.6-luna list prices, USD per million tokens (third-party listings, Sept 2026)
PRICE_INPUT, PRICE_CACHED, PRICE_OUTPUT = 0.20, 0.02, 1.20

LABELLING_HEADER = """PILOT {version} + {min_agree}-of-{runs} voting - extraction A ({edges} edges)
Label from THIS file - a fresh run renumbers the edges.
Chunks marked [HOLDOUT] were drawn at random after the prompt and checks were fixed.

LABEL ALL {edges} EDGES (E1-E{edges}). List the wrong ones.

LABELLING STANDARD (v5's, fixed for this iteration). An edge is WRONG if:
  1. The verb doesn't match its definition.
  2. The direction is reversed.
  3. The quote doesn't STATE the relation - a purpose, benefit or effect is
     inferred. "X for Y" and "access to Y" do not state that X contains Y.
  4. An endpoint isn't the real participant in the quote (a substitute, or the
     wrong actor).
  conforms_to = "based on or complies with"; a quote saying "based on" is correct.

"""


def find_chunk(source: str, chunk_index: int) -> dict:
    if source in FIXTURES:
        chunks = json.loads(FIXTURES[source].read_text())["chunks"]
    else:
        chunks = [c for c in store.metadata if c["filename"] == source]
    for chunk in chunks:
        if chunk["chunk_index"] == chunk_index:
            return chunk
    raise SystemExit(f"Chunk {chunk_index} of {source} was not found.")


def located(edge: dict) -> tuple:
    """An edge's identity across the whole pilot: which chunk, and what it says."""
    return (edge["doc_id"], edge["chunk_index"], *edge_key(edge))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def print_chunk(number: int, chunk: dict, trap: str, result: dict) -> None:
    print("=" * 78)
    holdout = "  [HOLDOUT]" if result["group"] == "holdout" else ""
    print(f"[{number}] {chunk['filename']}  chunk {chunk['chunk_index']}  "
          f"pages {chunk['page_start']}-{chunk['page_end']}  ocr={chunk['ocr']}{holdout}")
    print(f"    tests: {trap}")
    print("-" * 78)
    print(textwrap.indent(clean(chunk["text"]), "    | "))
    print("-" * 78)

    if "error" in result:
        print(f"    EXTRACTION FAILED: {result['error']}")
        return

    print(f"ENTITIES ({len(result['entities'])} kept)")
    for e in result["entities"]:
        print(f"    {e['type']:<10} {e['name']}")

    print(f"\nEDGES ({len(result['relations'])} kept)")
    for r in result["relations"]:
        print(f"    {r['id']:<4} {r['subject']}  --{r['relation']}-->  {r['object']}")
        print(textwrap.fill(f'"{r["evidence"]}"', width=78, break_on_hyphens=False,
                            initial_indent="           ", subsequent_indent="           "))

    if result["dropped"]:
        print(f"\nDROPPED (across all {result['runs']} runs)")
        for d in result["dropped"]:
            if d["kind"] == "entity":
                print(f"    entity    {d['reason']:<22} {d['type']}: {d['name']}")
            else:
                print(f"    relation  {d['reason']:<22} {d['subject']} --{d['relation']}--> {d['object']}")
                if d["reason"] != "unstable":
                    print(textwrap.fill(f'evidence: "{d["evidence"]}"', width=78, break_on_hyphens=False,
                                        initial_indent="              ", subsequent_indent="              "))
    print()


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} = {part / whole:.0%}" if whole else "n/a"


def check(value: float, target: float, higher_is_better: bool = True) -> str:
    ok = value >= target if higher_is_better else value <= target
    sign = ">=" if higher_is_better else "<="
    return f"target {sign} {target:.0%}   {'PASS' if ok else 'FAIL'}"


BOUNDARY = ("creates", "produces")   # two verbs whose definitions may overlap


def boundary_votes(results: list[dict]) -> dict:
    """
    Votes for each BOUNDARY verb, per (chunk, subject, object), counting every
    run that proposed it — kept edges and unstable ones alike.
    """
    votes = defaultdict(Counter)
    for r in results:
        if "error" in r:
            continue
        for e in r["relations"] + [d for d in r["dropped"] if d["reason"] == "unstable"]:
            if e["relation"] in BOUNDARY:
                subject, verb, obj = edge_key(e)
                votes[(r["chunk_index"], subject, obj)][verb] += e["votes"]
    return votes


def summarise(first: list[dict], second: list[dict]) -> None:
    """Quality numbers for extractions A (`first`) and B (`second`)."""
    ok = [r for r in first if "error" not in r]
    if not ok:
        print("Every extraction failed.")
        return

    kept = [e for r in ok for e in r["relations"]]
    unstable = [d for r in ok for d in r["dropped"] if d["reason"] == "unstable"]
    reasons = Counter(d["reason"] for r in ok for d in r["dropped"] if d["kind"] == "relation")
    proposed = sum(r["proposed_relations"] for r in ok)
    quoted = sum(r["quoted_relations"] for r in ok)

    # Every validated edge in every single run, counted once per run that found it
    raw = Counter()
    for e in kept + unstable:
        raw[e["relation"]] += e["votes"]
    raw_total = sum(raw.values())

    related = sum(1 for e in kept if e["relation"] == "related_to")

    # A systematic error found in v2, counted from the model's own entity types.
    types = {(e["doc_id"], e["chunk_index"], match_form(e["name"])): e["type"]
             for r in ok for e in r["entities"]}
    object_type = lambda e: types.get((e["doc_id"], e["chunk_index"], match_form(e["object"])))
    protects_threat = sum(1 for e in kept if e["relation"] == "protects" and object_type(e) == "threat")

    # Stability, on the known chunks only so it stays comparable with earlier
    # versions. Raw: every pair of single runs inside A. Voted: A against B.
    known = [r for r in ok if r["group"] == "known"]
    run_sets = [set() for _ in range(EXTRACT_RUNS)]
    for r in known:
        for i, keys in enumerate(r["run_edges"]):
            run_sets[i] |= {(r["doc_id"], r["chunk_index"], *k) for k in keys}
    run_pairs = list(itertools.combinations(run_sets, 2))
    raw_jaccard = sum(jaccard(a, b) for a, b in run_pairs) / len(run_pairs)
    first_edges = {located(e) for r in known for e in r["relations"]}
    second_edges = {located(e) for r in second if "error" not in r and r["group"] == "known"
                    for e in r["relations"]}
    voted_jaccard = jaccard(first_edges, second_edges)

    unanimous = [e for e in kept if e["votes"] == EXTRACT_RUNS]
    majority = [e for e in kept if e["votes"] < EXTRACT_RUNS]
    splits = [(r, s) for r in ok for s in r["verb_splits"]]

    print("=" * 78)
    print(f"SUMMARY  ({len(ok)} chunks, prompt {GRAPH_PROMPT_VERSION}, "
          f"edge kept if {EXTRACT_MIN_AGREE} of {EXTRACT_RUNS} runs find it)")
    print("-" * 78)
    print(f"  entities kept                 {sum(len(r['entities']) for r in ok)}")
    print(f"  edges kept                    {len(kept)}   from {proposed} proposals across all runs")
    print(f"    found by all {EXTRACT_RUNS} runs          {len(unanimous)}")
    print(f"    found by {EXTRACT_MIN_AGREE} of {EXTRACT_RUNS} only          {len(majority)}   "
          f"{' '.join(e['id'] for e in majority)}")
    print(f"  relation drops by reason      {dict(reasons.most_common()) or 'none'}")
    print(f"  verbs, kept edges             {dict(Counter(e['relation'] for e in kept).most_common())}")
    print()
    print(f"  quote found in the chunk      {pct(quoted, proposed):<18}"
          f"{check(quoted / proposed if proposed else 0, TARGET_QUOTED)}")
    print(f"  rejected: quote not found     {reasons['unquoted']}")
    print(f"  rejected: quote doesn't name  {reasons['endpoint_not_in_quote']}")
    print(f"            both ends")
    print(f"  rejected: nested name         {reasons['nested_name']}")
    print(f"  rejected: generic modifier    {reasons['generic_modifier']}")
    print(f"  rejected: negated             {reasons['negated']}")
    print(f"  rejected: verb not stated     {reasons['verb_not_stated']}")
    print(f"  related_to, kept edges        {pct(related, len(kept)):<18}"
          f"{check(related / len(kept) if kept else 0, TARGET_RELATED_TO, higher_is_better=False)}")
    print(f"  related_to, every single run  {pct(raw['related_to'], raw_total)}")
    print(f"  protects -> a threat          {protects_threat}   (should be 0: use mitigates)")
    print()
    print(f"  single-run agreement (known)  {raw_jaccard:<18.2f}({BASE_LABEL}: {BASE_RAW})")
    print(f"  voted agreement A vs B (known){voted_jaccard:<18.2f}({BASE_LABEL}: {BASE_VOTED})   "
          f"{check(voted_jaccard, TARGET_JACCARD)}")
    for label, diff in (("only in A", first_edges - second_edges), ("only in B", second_edges - first_edges)):
        for _, index, subject, relation, obj in sorted(diff):
            print(f"      {label}: chunk {index}  {subject} --{relation}--> {obj}")
    print()
    print(f"  VERB SPLITS in A: {len(splits)} entity pairs that {EXTRACT_MIN_AGREE}+ runs connected,"
          f" but under no verb with {EXTRACT_MIN_AGREE} votes")
    for r, s in splits:
        print(f"      chunk {r['chunk_index']}  "
              f"{' | '.join(f'{edge} (x{n})' for edge, n in s['votes'].items())}")
    print()

    # Where the creates / produces boundary was ambiguous: the same two
    # entities got both verbs, from different runs of A or from A versus B.
    in_a, in_b = boundary_votes(first), boundary_votes(second)
    pairs = in_a.keys() | in_b.keys()
    ambiguous = sorted(k for k in pairs if len(set(in_a[k]) | set(in_b[k])) > 1)
    tally = lambda votes: ", ".join(f"{v} x{n}" for v, n in sorted(votes.items())) or "-"
    print(f"  CREATES vs PRODUCES: {len(ambiguous)} of {len(pairs)} entity pairs that got either verb"
          f" got both (all {2 * EXTRACT_RUNS} runs)")
    for index, subject, obj in ambiguous:
        print(f"      chunk {index}  {subject} -> {obj}   "
              f"A: {tally(in_a[(index, subject, obj)])}   B: {tally(in_b[(index, subject, obj)])}")
    print()


def summarise_groups(first: list[dict]) -> None:
    """Known chunks against holdout chunks: the numbers that show whether the fixes generalise."""
    print("=" * 78)
    print("KNOWN vs HOLDOUT  (extraction A; precision needs your labels)")
    print("-" * 78)
    print(f"  {'':<34}{'known':>10}{'holdout':>10}")
    rows = {g: [r for r in first if "error" not in r and r["group"] == g] for g in GROUPS}
    count = lambda g, f: sum(f(r) for r in rows[g])
    for label, f in (
        ("chunks", lambda r: 1),
        ("edges kept", lambda r: len(r["relations"])),
        ("proposals, all runs", lambda r: r["proposed_relations"]),
        ("quote found in the chunk", lambda r: r["quoted_relations"]),
        ("rejected: quote not found", lambda r: sum(d["reason"] == "unquoted" for d in r["dropped"])),
        ("rejected: verb not stated", lambda r: sum(d["reason"] == "verb_not_stated" for d in r["dropped"])),
    ):
        print(f"  {label:<34}{count('known', f):>10}{count('holdout', f):>10}")
    edge_ids = {g: " ".join(e["id"] for r in rows[g] for e in r["relations"]) for g in GROUPS}
    for g in GROUPS:
        print(f"  {g} edges: {edge_ids[g] or '-'}")
    print()


def print_verb_not_stated(first: list[dict]) -> None:
    """
    Every edge the verb-word check rejected in A, so correct edges it throws
    away can be audited. The same edge rejected in several runs is listed once.
    """
    rejected = defaultdict(list)            # (group, chunk, edge key) -> the rejected proposals
    for r in first:
        for d in r.get("dropped", []):
            if d["reason"] == "verb_not_stated":
                rejected[(r["group"], r["chunk_index"], edge_key(d))].append(d)
    print("=" * 78)
    print(f"VERB NOT STATED: {len(rejected)} distinct edges rejected in extraction A "
          f"({sum(len(v) for v in rejected.values())} proposals across its runs)")
    print("-" * 78)
    for (group, index, _), proposals in sorted(rejected.items(), key=lambda item: (GROUPS.index(item[0][0]), item[0][1])):
        d = proposals[0]
        print(f"  [{group}] chunk {index}  {d['subject']} --{d['relation']}--> {d['object']}"
              f"   ({len(proposals)} of {EXTRACT_RUNS} runs)")
        print(textwrap.fill(f'"{d["evidence"]}"', width=78, break_on_hyphens=False,
                            initial_indent="        ", subsequent_indent="        "))
    print()


EDGE_FIELDS = ("id", "chunk_index", "subject", "relation", "object", "votes")


def save_run(first: list[dict], second: list[dict]) -> Path:
    """Keep this run's edges, so a later run can compare verbs and labels can be scored."""
    def edges(results: list[dict]) -> list[dict]:
        return [{"group": r["group"], **{k: e[k] for k in EDGE_FIELDS if k in e}}
                for r in results if "error" not in r for e in r["relations"]]

    saved_at = datetime.now(timezone.utc)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{GRAPH_PROMPT_VERSION}-{saved_at:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps({
        "prompt_version": GRAPH_PROMPT_VERSION,
        "saved_at": saved_at.isoformat(),
        "verbs": list(typing.get_args(RelationType)),
        "chunks": [{"source": s, "chunk_index": i, "group": g} for s, i, g, _ in PILOT],
        "a": edges(first),
        "b": edges(second),
    }, indent=2))
    return path


def previous_run() -> dict | None:
    """The newest saved run from an EARLIER prompt version, if there is one."""
    runs = [json.loads(p.read_text()) for p in RUNS_DIR.glob("*.json")]
    earlier = [r for r in runs if r["prompt_version"] != GRAPH_PROMPT_VERSION]
    return max(earlier, key=lambda r: r["saved_at"]) if earlier else None


def print_verb_changes(first: list[dict], previous: dict | None) -> None:
    """
    Edges whose two entities were connected by a different verb in the previous
    prompt version. The pattern to watch is a NEW verb taking an edge that
    already had a specific verb; a new verb taking a related_to edge is the
    point of adding it.
    """
    print("=" * 78)
    if previous is None:
        print("VERB CHANGES: no saved run from an earlier prompt version to compare with.\n")
        return

    before = defaultdict(set)           # (chunk, entity pair) -> verbs in the previous run's A
    for e in previous["a"]:
        before[(e["chunk_index"], frozenset((match_form(e["subject"]), match_form(e["object"]))))].add(e["relation"])
    new_verbs = set(typing.get_args(RelationType)) - set(previous["verbs"])

    changes = []
    for r in first:
        if r.get("group") != "known":   # the previous run only covered the known chunks
            continue
        for e in r.get("relations", []):
            old = before.get((e["chunk_index"], frozenset((match_form(e["subject"]), match_form(e["object"])))))
            if old and e["relation"] not in old:
                if e["relation"] in new_verbs and old != {"related_to"}:
                    kind = "NEW VERB took an edge that already had a verb"
                elif old == {"related_to"}:
                    kind = "from related_to"
                else:
                    kind = "changed"
                changes.append((e, old, kind))

    watch = sum(1 for *_, kind in changes if kind.startswith("NEW VERB"))
    print(f"VERB CHANGES since {previous['prompt_version']} (run saved {previous['saved_at'][:16]}): "
          f"{len(changes)} edges, {watch} of them a new verb taking an edge that had a verb")
    print(f"  verbs new in {GRAPH_PROMPT_VERSION}: {', '.join(sorted(new_verbs)) or 'none'}")
    for e, old, kind in changes:
        print(f"      {e['id']:<4} chunk {e['chunk_index']:>3}  {e['subject']} -> {e['object']}   "
              f"{'/'.join(sorted(old))} -> {e['relation']}   {kind}")
    print()


async def main():
    chunks = [find_chunk(source, index) for source, index, _, _ in PILOT]
    first = await extract_many(chunks)
    second = await extract_many(chunks)

    # Tag every result with its chunk and group, and number A's edges once,
    # so the chunk listing, the summary and the saved run use the same ids.
    ids = (f"E{n}" for n in itertools.count(1))
    for results in (first, second):
        for chunk, (_, _, group, _), result in zip(chunks, PILOT, results):
            result["doc_id"], result["chunk_index"], result["group"] = chunk["doc_id"], chunk["chunk_index"], group
    for result in first:
        for r in result.get("relations", []):
            r["id"] = next(ids)

    # The chunk listing is printed and also written out as the file to label.
    listing = io.StringIO()
    with contextlib.redirect_stdout(listing):
        for number, (chunk, (_, _, _, trap), result) in enumerate(zip(chunks, PILOT, first), start=1):
            print_chunk(number, chunk, trap, result)
    print(listing.getvalue(), end="")
    edge_count = sum(len(r.get("relations", [])) for r in first)
    labelling = Path(f"eval/pilot_{GRAPH_PROMPT_VERSION}_for_labelling.txt")
    labelling.write_text(LABELLING_HEADER.format(version=GRAPH_PROMPT_VERSION, min_agree=EXTRACT_MIN_AGREE,
                                                 runs=EXTRACT_RUNS, edges=edge_count) + listing.getvalue())

    summarise(first, second)
    summarise_groups(first)
    print_verb_not_stated(first)
    print_verb_changes(first, previous_run())
    saved = save_run(first, second)

    # --- cost: measured on extraction A, projected to the whole corpus -
    ok = [r for r in first if "error" not in r]
    tokens = Counter()
    for r in ok:
        tokens.update(r["usage"])
    calls = len(ok) * EXTRACT_RUNS
    total = store.count()
    uncached = tokens["input"] - tokens["cached"]
    cost = (uncached * PRICE_INPUT + tokens["cached"] * PRICE_CACHED
            + tokens["output"] * PRICE_OUTPUT) / 1e6

    print("=" * 78)
    print(f"TOKENS  (extraction A: {len(ok)} chunks x {EXTRACT_RUNS} runs = {calls} calls)")
    print("-" * 78)
    print(f"  input {tokens['input']:,}  (cached {tokens['cached']:,})   output {tokens['output']:,}")
    print(f"  per call: input {tokens['input'] / calls:,.0f}   output {tokens['output'] / calls:,.0f}")
    print(f"  cost of extraction A:              ${cost:.4f}")
    print(f"  projected for all {total} chunks:    ${cost / len(ok) * total:.2f}")
    print()
    print(f"Edges, ids and votes saved to {saved}")
    print(f"Label from {labelling} - it has no vote counts.")


if __name__ == "__main__":
    asyncio.run(main())
