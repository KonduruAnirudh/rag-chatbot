# app/rag/graph_retrieve.py
"""
Step 8 (EXPERIMENTAL): graph evidence for a question, to sit beside vector retrieval.

Step 7 measured this and found it does NOT improve retrieval on this corpus
(README, "Graph retrieval experiment"). It exists to demonstrate the architecture
end to end, not because it helps.

The method is Step 7's G1, chosen on the development half by the pre-registered
rule, copied here unchanged because production code does not import from eval/.
eval/step8_check.py verifies it links and selects exactly what Step 7 did.

  1. link   the admitted entities the question names: every name the graph knows
            for them, as whole words with an optional plural, longest match first
  2. reach  the chunks whose quote supports an edge touching a linked entity (1 hop)
  3. gate   a chunk counts only if it passes the production cosine gate and is not
            already one of the vector chunks
  4. rank   reached from the most linked entities first, then cosine
  5. cap    at most GRAPH_EVIDENCE_MAX chunks

Refusal is never weakened: with no vector chunks this returns nothing before
looking at the graph, so graph evidence can only ever sit beside vector evidence.

Relation verbs play no part in choosing evidence. They are kept in the provenance
for people to read; the model is never shown them (they are about 48% correct).
"""
import re
from collections import defaultdict

from app.config import GRAPH_EVIDENCE_MAX, SIMILARITY_THRESHOLD
from app.rag.embed import embed_query
from app.rag.graph import graph_store
from app.rag.graph_extract import match_form
from app.rag.store import store

DIAGRAM_MARK = "[diagram"     # vision OCR's own description of an image: machine-written, weaker evidence


# --- 1. linking (identical to eval/linking_recall.py) -------------------------

def entity_forms(graph) -> dict[str, str]:
    """Every name a question could use for an admitted entity -> that entity."""
    forms = {}
    for key in sorted(graph.admitted):
        surfaces = graph.entities.get(key, {}).get("surface_forms", {})
        for form in {key, *(match_form(s) for s in surfaces)}:
            forms.setdefault(form, key)
    return forms


def link(question: str, forms: dict[str, str]) -> list[str]:
    """Entities named in the question: longest matches first, no overlapping matches."""
    text = match_form(question)
    found = []
    for form, key in forms.items():
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(form) + r"(?:e?s)?(?![a-z0-9])", text):
            found.append((m.start(), m.end(), key))
    found.sort(key=lambda span: (span[0] - span[1], span[0]))          # longest first
    taken, keys = [], []
    for start, end, key in found:
        if all(end <= a or start >= b for a, b in taken):
            taken.append((start, end))
            keys.append(key)
    return list(dict.fromkeys(keys))


# --- 2. reach -------------------------------------------------------------------

_index = {"graph": None}


def graph_index(graph) -> dict:
    """Names and edge lookups for one graph. Every upload or delete swaps in a new graph, which rebuilds this."""
    if _index["graph"] is not graph:
        edges = defaultdict(list)          # {a, b} -> [(subject, verb, object, provenance), ...]
        for (subject, verb, obj), provenance in graph.edges.items():
            edges[frozenset((subject, obj))].append((subject, verb, obj, provenance))
        _index.update(graph=graph, forms=entity_forms(graph), edges=edges)
    return _index


def reach(seeds: list[str], graph, edges: dict) -> dict:
    """Chunk (doc_id, chunk_index) -> the linked entities that reach it, and the edges quoted in it that do."""
    found = defaultdict(lambda: {"seeds": set(), "links": []})
    for seed in seeds:
        for neighbour in graph.adjacency.get(seed, set()):
            for subject, verb, obj, provenance in edges[frozenset((seed, neighbour))]:
                for p in provenance:
                    chunk = found[(p["doc_id"], p["chunk_index"])]
                    chunk["seeds"].add(seed)
                    chunk["links"].append((seed, neighbour, subject, verb, obj, p))
    return found


# --- 3-5. gate, rank, cap -----------------------------------------------------------

def provenance(found_chunk: dict, graph) -> dict:
    """What reached a chunk, in the entities' own names: the path, the edge, and the sentence it was quoted from."""
    name = lambda key: graph.entities[key]["canonical_name"]
    links = [{
        "reached_from": name(seed),
        "to": name(neighbour),
        "subject": name(subject),
        "relation": verb,                              # for people; never shown to the model
        "object": name(obj),
        "quote": p["evidence"],
        "subject_as_written": p["subject_as_written"],
        "object_as_written": p["object_as_written"],
        "votes": p["votes"],
        "from_diagram": DIAGRAM_MARK in p["evidence"].lower(),
    } for seed, neighbour, subject, verb, obj, p in found_chunk["links"]]
    links.sort(key=lambda l: (l["reached_from"], l["to"], l["relation"], l["quote"]))
    return {"linked_entities": sorted(name(s) for s in found_chunk["seeds"]), "links": links}


async def graph_evidence(question: str, vector_chunks: list[dict]) -> list[dict]:
    """
    Up to GRAPH_EVIDENCE_MAX chunks the graph reaches from the question's entities,
    each shaped like a vector chunk plus `retrieval` and `graph` (its provenance).
    Empty when the vector side found nothing, the graph is empty, or nothing links.
    """
    if not vector_chunks:
        return []                  # refusal stays structural: the graph never supplies the only evidence
    graph = graph_store.graph      # one reference for the whole request
    if graph is None:
        return []
    index = graph_index(graph)
    seeds = link(question, index["forms"])
    if not seeds:
        return []

    found = reach(seeds, graph, index["edges"])
    position = {(c["doc_id"], c["chunk_index"]): i for i, c in enumerate(store.metadata)}
    taken = {(c["doc_id"], c["chunk_index"]) for c in vector_chunks}
    # A second embedding call: retrieve() does not expose its scores, and the vector path stays untouched
    cosine = store.dense_scores(await embed_query(question))
    eligible = [chunk for chunk in found
                if chunk in position                       # skip a chunk whose document is mid-delete
                and chunk not in taken
                and cosine[position[chunk]] >= SIMILARITY_THRESHOLD]
    eligible.sort(key=lambda chunk: (len(found[chunk]["seeds"]), float(cosine[position[chunk]])), reverse=True)

    return [{
        **store.metadata[position[chunk]],
        "score": float(cosine[position[chunk]]),
        "retrieval": "graph",
        "graph": provenance(found[chunk], graph),
    } for chunk in eligible[:GRAPH_EVIDENCE_MAX]]
