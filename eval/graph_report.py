# eval/graph_report.py
"""
Step 6a report: what entity resolution does to the graph.

  1. merges by layer, and every entity that more than one name resolved to
  2. over-merge checks — names that must stay apart, and names that must join
  3. how much resolution changes the graph's shape
  4. layer-3 proposals, written out for a person to label (nothing is merged)

Reads data/graph/extractions.json; writes only the layer-3 labelling files.
The layer-3 proposals need the embeddings API (a fraction of a cent).

Run from the project root:
    python -m eval.graph_report
"""
import asyncio
import json
from collections import Counter
from pathlib import Path

from app.rag.graph import build_graph, graph_store, normalise, propose_merges, resolve

PROPOSALS_JSON = Path("eval/graph_l3_proposals.json")
PROPOSALS_TO_LABEL = Path("eval/graph_l3_proposals_for_labelling.txt")

# (names that must resolve to DIFFERENT entities, names that must resolve to ONE)
OVER_MERGE_CHECKS = {
    "Shield": (
        ["AWS Shield", "AWS Shield Standard", "AWS Shield Advanced",
         "AWS Shield Response Team (SRT)", "Origin Shield"],
        [["AWS Shield", "Shield"], ["AWS Shield Advanced", "Shield Advanced"],
         ["AWS Shield Response Team (SRT)", "AWS Shield Response Team (AWS SRT)", "SRT"]],
    ),
    "Security Hub": (
        ["Security Hub", "Security Hub CSPM"],
        [["AWS Security Hub CSPM", "Security Hub CSPM"]],
    ),
    "SET": (
        ["SET", "SET Recipient", "SET Transmitter", "SET Issuer", "SET Transmission Request"],
        [["SET", "Security Event Token (SET)"]],
    ),
    # Found by the first run of this report: stripping "AWS" from a common noun
    # merged the whitepaper's AWS endpoint with the RFC's HTTP endpoint.
    "Vendor prefix on a common noun": (
        ["AWS endpoint", "endpoint"],
        [["AWS WAF", "WAF"], ["Amazon CloudFront", "CloudFront"]],
    ),
    # Service vs a resource created with it; kept apart by a layer-2 decision.
    "VPC": (
        ["Amazon VPC", "virtual private cloud (VPC)"],
        [["Amazon VPC", "Amazon Virtual Private Cloud"]],
    ),
}


def shape_row(label: str, graph) -> str:
    s, paths = graph.stats(), graph.two_hop_paths()
    return (f"  {label:<34}{s['entities_resolved']:>9}{len(graph.admitted):>9}{s['nodes']:>7}{s['edges']:>7}"
            f"{s['components']:>7}{s['largest_component']:>9}{s['nodes_with_2_plus_neighbours']:>7}"
            f"{paths['cross_chunk']:>9}{paths['cross_document']:>7}")


def check_over_merges(resolution: dict) -> None:
    corpus_key = {name: key for name, (key, _) in resolution.items()}
    acronyms = {}
    for name in resolution:
        key, acronym = normalise(name)
        if acronym:
            acronyms.setdefault(acronym, key)
    def key_of(name):                        # names absent from the corpus are resolved by the same rules
        if name in corpus_key:
            return corpus_key[name], "corpus"
        key = normalise(name)[0]
        return acronyms.get(key, key), "not in corpus - rule applied"

    print("=" * 78)
    print("2. OVER-MERGE CHECKS")
    print("-" * 78)
    for family, (must_differ, must_join) in OVER_MERGE_CHECKS.items():
        keys = {n: key_of(n) for n in must_differ}
        distinct = len({k for k, _ in keys.values()}) == len(must_differ)
        absent = [n for n, (_, source) in keys.items() if source != "corpus"]
        if not distinct:
            status = "FAIL: OVER-MERGE"
        elif absent:
            # the rules keep the names apart, but the corpus never tested it
            status = f"UNTESTED - not in the corpus: {', '.join(absent)}"
        else:
            status = "PASS"
        print(f"  {family}: must stay apart -> {status}")
        for name, (key, source) in keys.items():
            print(f"      {name:<36} -> {key!r}" + ("" if source == "corpus" else f"   ({source})"))
        for group in must_join:
            joined = len({key_of(n)[0] for n in group}) == 1
            print(f"    must join {group}: {'PASS' if joined else 'not joined (under-merge)'}")
    print()


async def main():
    records = graph_store.records.values()
    texts = graph_store.texts()
    graphs = {level: build_graph(records, texts, level) for level in ("exact", "layer1", "layer2")}
    real = graphs["layer2"]

    # 1. merges -------------------------------------------------------------------
    names = {m["name"] for r in records for m in r["entities"]}
    resolution, ambiguous = resolve(names, "layer2")
    print("=" * 78)
    print(f"1. MERGES  ({len(names)} names -> {len({k for k, _ in resolution.values()})} entities)")
    print("-" * 78)
    layers = Counter(layer for e in real.entities.values() for layer in e["merged_by"].values())
    print(f"  how each name got its key: {dict(layers)}")
    print(f"    exact = case and spacing only; layer1 = prefix/article/brackets; acronym = a name the")
    print(f"    text defines in brackets; layer2 = hand-written decision; layer3-labelled = a similarity")
    print(f"    proposal a person labelled SAME")
    merged = {k: e for k, e in real.entities.items()
              if len({n.casefold() for n in e["surface_forms"]}) > 1}
    print(f"  entities that more than one name resolved to: {len(merged)}")
    for key, entity in sorted(merged.items()):
        docs = {doc for doc, _, _ in entity["mentions"]}
        forms = ", ".join(f"{n} [{entity['merged_by'][n]}]" for n in sorted(entity["surface_forms"]))
        print(f"    {key:<34} <- {forms}" + ("   ACROSS DOCUMENTS" if len(docs) > 1 else ""))
    print(f"  acronyms the text defines for more than one name (not used): {ambiguous or 'none'}")
    print(f"  edges that became self-loops after merging (dropped): {real.report['self_loops_from_merges']}")
    print(f"  admission: {len(real.admitted)} admitted; rejected "
          f"{dict(Counter(real.rejected.values()))}")
    print()

    # 2. over-merges ------------------------------------------------------------------
    check_over_merges(resolution)

    # 4. layer-3 proposals (computed before 3, whose last row uses them) -----------------
    proposals = await propose_merges(real)
    PROPOSALS_JSON.write_text(json.dumps(proposals, indent=2))
    lines = ["LAYER 3 PROPOSALS - label each pair: SAME thing (merge) or DIFFERENT (keep apart).",
             "Nothing has been merged. Under-merging is recoverable; over-merging makes the graph",
             "confidently wrong, so when unsure, answer DIFFERENT.", ""]
    for n, p in enumerate(proposals, start=1):
        lines.append(f"P{n:<3} [{p['type']}]  {p['name_a']}   |   {p['name_b']}")
    PROPOSALS_TO_LABEL.write_text("\n".join(lines) + "\n")

    # 3. shape ------------------------------------------------------------------------
    parent = {}
    def find(k):
        while parent.get(k, k) != k:
            k = parent[k]
        return k
    for p in proposals:
        parent[find(p["b"])] = find(p["a"])
    hypothetical = build_graph(records, texts, "layer2", extra_merges={k: find(k) for k in parent})

    print("=" * 78)
    print("3. SHAPE - how much resolution changes the graph")
    print("-" * 78)
    print(f"  {'':<34}{'entities':>9}{'admitted':>9}{'nodes':>7}{'edges':>7}{'comps':>7}{'largest':>9}"
          f"{'2+nbr':>7}{'x-chunk':>9}{'x-doc':>7}")
    print(shape_row("no resolution (exact names)", graphs["exact"]))
    print(shape_row("layer 1 (prefix, acronyms)", graphs["layer1"]))
    print(shape_row("+ layer 2, labelled L3 = THE GRAPH", real))
    print(shape_row(f"+ all {len(proposals)} L3 proposals (if accepted)", hypothetical))
    print("  x-chunk / x-doc = two-hop paths whose edges come from different chunks / documents")
    print("  The last row is an upper bound: it assumes every proposal is right, and some won't be.")
    print()
    # 5. hubs, by neighbour count (the definition agreed for Step 7) --------------------------
    paths = real.two_hop_paths()
    through, total = paths["cross_chunk_through"], paths["cross_chunk"]
    ranked = sorted(real.adjacency, key=real.neighbour_count, reverse=True)
    print("=" * 78)
    print("5. HUBS by neighbour count - how much traversal depends on a few nodes")
    print("-" * 78)
    for top in (1, 3, 10):
        share = sum(through[k] for k in ranked[:top]) / total if total else 0
        print(f"  cross-chunk two-hop paths through the top {top:>2} node(s): {share:.0%}")
    print(f"  {'entity':<34}{'neighbours':>11}{'paths through':>15}")
    for key in ranked[:12]:
        print(f"  {real.entities[key]['canonical_name'][:33]:<34}{real.neighbour_count(key):>11}{through[key]:>15}")
    print()

    print("=" * 78)
    print(f"4. LAYER-3 PROPOSALS: {len(proposals)} pairs (cosine >= 0.90, same type, qualifier and")
    print(f"   number guards passed). To label: {PROPOSALS_TO_LABEL}  (cosines hidden; they are in")
    print(f"   {PROPOSALS_JSON})")


if __name__ == "__main__":
    asyncio.run(main())
