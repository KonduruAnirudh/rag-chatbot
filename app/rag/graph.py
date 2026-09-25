# app/rag/graph.py
"""
The knowledge graph, derived from extracted entity and relation records.

SOURCE OF TRUTH: data/graph/extractions.json — the raw per-chunk records from
graph_extract, keyed "doc_id:chunk_index". Everything else here is DERIVED from
them when the store loads, and is never saved: a derived index that is never
stored can never disagree with its source (the same principle as BM25 in store.py).

Deriving the graph, in order:
  1. RESOLUTION — one key per real-world thing
       layer 1  normalise: Unicode, case, a leading "the", an "AWS"/"Amazon" prefix
                when what follows is itself a name ("AWS WAF", not "AWS endpoint").
                A trailing "(ACRONYM)" whose letters are the name's initials is
                harvested as an alias. Qualifiers ("Advanced", "Standard", "CSPM")
                are never stripped, so Shield and Shield Advanced stay apart.
       layer 2  hand-written decisions (graph_aliases.json), keyed by the name as
                written and applied before any automatic rule: they can join names
                the text never links, or keep apart names the rules would merge
       layer 3  embedding similarity — only PROPOSES merges for a person to label
                (propose_merges); it never merges anything on its own
  2. ADMISSION — which entities become nodes
       R2  its name is not a generic word ("resource", "user", ...)
       R3  named (capitalised mid-sentence, an acronym, a digit or a symbol) or
           recurring (named in 2+ distinct sentences of the corpus)
       R1  it has an edge to another admitted entity — re-applied until nothing
           changes, because removing one entity can strand another
  3. EDGES — one per (subject, verb, object); each chunk that states it adds a
       provenance record carrying its verbatim quote

Hubs are never removed. neighbour_count() lets traversal decide how to treat
them; that policy is measured in Step 7.
"""
import json
import re
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np

from app.rag.graph_extract import GENERIC_WORDS, clean, match_form

GRAPH_DIR = Path("data/graph")
EXTRACTIONS_PATH = GRAPH_DIR / "extractions.json"
ALIASES_PATH = Path(__file__).with_name("graph_aliases.json")

VENDOR_PREFIXES = ("aws ", "amazon ")
KEEP_PREFIX = {"support", "config", "organizations"}   # "AWS Support" is a product; "support" is a word
_STOPWORDS = {"of", "and", "the", "for", "a", "an", "to", "on", "in"}
_TRAILING_ACRONYM = re.compile(r"^(?P<base>.*\S)\s*\((?P<acronym>[^()]+)\)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

LEVELS = ("exact", "layer1", "layer2")   # how much resolution to apply; "layer2" is the real graph


# --- resolution ---------------------------------------------------------

def _strip_vendor(key: str) -> str:
    for prefix in VENDOR_PREFIXES:
        rest = key[len(prefix):]
        if key.startswith(prefix) and rest and rest not in KEEP_PREFIX:
            return rest
    return key


def _initials(phrase: str) -> set[str]:
    """
    The phrase's possible initials. Acronyms disagree on hyphens and small
    words: "DNS-Based Authentication of Named Entities (DANE)" counts the
    hyphenated word once and skips "of"; "X-Forwarded-For (XFF)" counts every
    part, "for" included. The acronym sitting in brackets right after the name
    is the real evidence; this is only a check that its letters fit.
    """
    found = set()
    for separators in (r"[\s/]+", r"[\s\-/]+"):
        for skip_small_words in (True, False):
            words = [w for w in re.split(separators, phrase)
                     if w and not (skip_small_words and w in _STOPWORDS)]
            if words and words[0] in ("aws", "amazon"):
                words = words[1:]
            found.add("".join(w[0] for w in words))
    return found


def normalise(name: str) -> tuple[str, str | None]:
    """
    Layer 1. Returns (key, acronym this name defines or None).

    "AWS Shield Response Team (AWS SRT)" -> ("shield response team", "srt")
    "Network Load Balancer (NLBs)"       -> ("network load balancer", "nlb")
    "HTTP Response Status Code 202 (Accepted)" keeps its brackets: "accepted"
    is not the name's initials, so it is not an acronym.

    The vendor prefix goes only when what follows is itself a name, as
    written: "AWS WAF" -> "waf", but "AWS endpoint" stays whole — stripped, it
    would merge with every other endpoint, including the RFC's.
    """
    key = match_form(name).strip(" \"'.,;:")
    if key.startswith("the "):
        key = key[4:]
    words = clean(name).split()
    if words and words[0].casefold() == "the":
        words = words[1:]
    prefix_names_a_product = len(words) > 1 and (words[1][:1].isupper() or words[1][:1].isdigit())
    acronym = None
    match = _TRAILING_ACRONYM.match(key)
    if match:
        inner = _strip_vendor(match["acronym"].strip())
        for candidate in (inner, inner[:-1] if inner.endswith("s") else None):
            if candidate and re.fullmatch(r"[a-z0-9]{2,8}", candidate) and candidate in _initials(match["base"]):
                key, acronym = match["base"], candidate
                break
    return (_strip_vendor(key) if prefix_names_a_product else key), acronym


def load_decisions() -> dict:
    """
    Hand-written resolution decisions from graph_aliases.json:
      aliases     name as written -> key (layer 2, applied before the automatic rules)
      merges      key -> key, layer-3 proposals a person labelled SAME (applied after them)
      keep_apart  pairs a person labelled DIFFERENT; never proposed again
    """
    data = json.loads(ALIASES_PATH.read_text()) if ALIASES_PATH.exists() else {}
    return {"aliases": data.get("aliases", {}), "merges": data.get("merges", {}),
            "keep_apart": [frozenset(pair) for pair in data.get("keep_apart", [])]}


def load_aliases() -> dict[str, str]:
    return load_decisions()["aliases"]


def resolve(names, level: str = "layer2", aliases: dict | None = None) -> tuple[dict, dict]:
    """
    Map every surface form to (key, how it got there).

    Returns ({name: (key, layer)}, {acronym: keys}) — the second holds acronyms
    the text defines for MORE than one name; they are not used, and are
    reported instead of guessed.
    """
    if level == "exact":
        return {n: (match_form(n), "exact") for n in names}, {}

    first = {n: normalise(n) for n in names}
    defined = defaultdict(set)
    for key, acronym in first.values():
        if acronym:
            defined[acronym].add(key)
    acronyms = {a: next(iter(keys)) for a, keys in defined.items() if len(keys) == 1}
    ambiguous = {a: sorted(keys) for a, keys in defined.items() if len(keys) > 1}
    hand = (aliases if aliases is not None else load_aliases()) if level == "layer2" else {}

    resolved = {}
    for name, (key, _) in first.items():
        written = match_form(name)
        if written in hand:                 # a person's decision overrides every automatic rule
            resolved[name] = (hand[written], "layer2")
            continue
        layer = "exact" if key == written else "layer1"
        if key in acronyms and acronyms[key] != key:
            key, layer = acronyms[key], "acronym"
        resolved[name] = (key, layer)
    return resolved, ambiguous


# --- admission helpers ----------------------------------------------------

def is_generic(key: str) -> bool:
    return key in GENERIC_WORDS or key.removesuffix("s") in GENERIC_WORDS


def _is_named(surface: str, chunk_text: str) -> bool:
    """Capitalised other than at a sentence start, or an acronym, digit or symbol."""
    words = surface.split()
    if re.search(r"[0-9/+.()@_]", surface):
        return True
    if any(sum(ch.isupper() for ch in w) >= 2 for w in words):          # acronym, CamelCase
        return True
    if any(w[:1].isupper() for w in words[1:]):
        return True
    if words and words[0][:1].isupper():                                # first word only: check the text
        text = " ".join(chunk_text.split())
        for match in re.finditer(re.escape(surface), text):
            before = text[:match.start()].rstrip()
            if before and before[-1] not in ".!?:\"'":
                return True
    return False


def _mentions(form: str, sentence: str) -> bool:
    return form in sentence and bool(
        re.search(r"(?<![a-z0-9])" + re.escape(form) + r"(?:e?s)?(?![a-z0-9])", sentence))


# --- the graph ----------------------------------------------------------------

class Graph:
    """One derived graph: admitted nodes, edges with provenance, and why the rest were left out."""

    def __init__(self, entities: dict, edges: dict, admitted: set, rejected: dict,
                 report: dict, candidates: set):
        self.entities = entities        # key -> {canonical_name, type, surface_forms, merged_by, mentions}
        self.edges = edges              # (subject, verb, object) -> [provenance, ...]  (admitted only)
        self.admitted = admitted
        self.rejected = rejected        # key -> reason
        self.report = report
        self.candidates = candidates    # every non-generic key that appears in an edge
        self.adjacency = defaultdict(set)
        for subject, _, obj in edges:
            self.adjacency[subject].add(obj)
            self.adjacency[obj].add(subject)

    def neighbour_count(self, key: str) -> int:
        return len(self.adjacency.get(key, ()))

    def neighbours(self, key: str) -> list[dict]:
        """Every edge touching `key`, in either direction, with its provenance."""
        return [{"subject": s, "relation": v, "object": o, "provenance": prov}
                for (s, v, o), prov in self.edges.items() if key in (s, o)]

    def components(self) -> list[set]:
        seen, found = set(), []
        for start in self.adjacency:
            if start in seen:
                continue
            component, queue = {start}, deque([start])
            seen.add(start)
            while queue:
                for nxt in self.adjacency[queue.popleft()]:
                    if nxt not in seen:
                        seen.add(nxt)
                        component.add(nxt)
                        queue.append(nxt)
            found.append(component)
        return sorted(found, key=len, reverse=True)

    def two_hop_paths(self) -> dict:
        """
        Paths a - b - c. A path matters for multi-hop retrieval when its two edges
        come from different chunks: vector search can return one chunk, not the link.
        """
        where = defaultdict(set)
        for (s, _, o), prov in self.edges.items():
            where[frozenset((s, o))] |= {(p["doc_id"], p["chunk_index"]) for p in prov}
        total = cross_chunk = cross_doc = 0
        through = Counter()                 # middle node -> cross-chunk paths that pass through it
        for middle, ends in self.adjacency.items():
            for a, c in combinations(sorted(ends), 2):
                total += 1
                first, second = where[frozenset((a, middle))], where[frozenset((middle, c))]
                if any(x != y for x in first for y in second):
                    cross_chunk += 1
                    through[middle] += 1
                cross_doc += any(x[0] != y[0] for x in first for y in second)
        return {"total": total, "cross_chunk": cross_chunk, "cross_document": cross_doc,
                "cross_chunk_through": through}

    def stats(self) -> dict:
        components = self.components()
        degrees = Counter(self.neighbour_count(k) for k in self.adjacency)
        return {
            "entities_resolved": len(self.entities),
            "nodes": len(self.adjacency),
            "edges": len(self.edges),
            "linked_pairs": len({frozenset((s, o)) for s, _, o in self.edges}),
            "nodes_with_2_plus_neighbours": sum(n for d, n in degrees.items() if d >= 2),
            "components": len(components),
            "largest_component": len(components[0]) if components else 0,
            "two_hop_paths": self.two_hop_paths(),
        }


def build_graph(records, texts: dict, level: str = "layer2",
                aliases: dict | None = None, extra_merges: dict | None = None) -> Graph:
    """
    Derive a graph from extraction records.

    texts        {(doc_id, chunk_index): chunk text}, for the named/recurring tests
    level        "exact", "layer1" or "layer2" — how much resolution to apply.
                 At "layer2", the layer-3 merges a person labelled SAME are applied too.
    extra_merges {key: key} applied after resolution; used only to measure what
                 accepting layer-3 proposals WOULD do, never to build the real graph
    """
    records = [r for r in records if "error" not in r]
    names = {m["name"] for r in records for m in r["entities"]}
    names |= {e[end] for r in records for e in r["relations"] for end in ("subject", "object")}
    resolution, ambiguous = resolve(names, level, aliases)
    labelled = load_decisions()["merges"] if level == "layer2" else {}
    merge = {**labelled, **(extra_merges or {})}
    for name, (key, layer) in resolution.items():
        if key in labelled:
            resolution[name] = (key, "layer3-labelled")
    key_of = lambda name: merge.get(resolution[name][0], resolution[name][0])

    # Entities: every surface form that resolved to the key, with where it was seen
    entities = defaultdict(lambda: {"surface_forms": Counter(), "types": Counter(),
                                    "merged_by": {}, "mentions": []})
    for r in records:
        for m in r["entities"]:
            entity = entities[key_of(m["name"])]
            entity["surface_forms"][m["name"]] += 1
            entity["types"][m["type"]] += 1
            entity["merged_by"][m["name"]] = resolution[m["name"]][1]
            entity["mentions"].append((m["doc_id"], m["chunk_index"], m["name"]))
    for key, entity in entities.items():
        forms = entity["surface_forms"]
        entity["canonical_name"] = max(forms, key=lambda f: (forms[f], len(f)))
        entity["type"] = entity.pop("types").most_common(1)[0][0]

    # Edges: one per (subject, verb, object), one provenance record per chunk stating it
    all_edges, self_loops = defaultdict(list), 0
    for r in records:
        for e in r["relations"]:
            subject, obj = key_of(e["subject"]), key_of(e["object"])
            if subject == obj:
                self_loops += 1          # two names for one thing, merged by resolution
                continue
            all_edges[(subject, e["relation"], obj)].append({
                "doc_id": e["doc_id"], "filename": e["filename"], "chunk_index": e["chunk_index"],
                "page_start": e["page_start"], "page_end": e["page_end"], "ocr": e["ocr"],
                "evidence": e["evidence"], "votes": e["votes"],
                "subject_as_written": e["subject"], "object_as_written": e["object"],
            })

    # Admission
    in_edges = {k for s, _, o in all_edges for k in (s, o)}
    rejected = {k: "generic word (R2)" for k in in_edges if is_generic(k)}
    candidates = in_edges - set(rejected)

    sentences = defaultdict(set)
    for (doc_id, _), text in texts.items():
        sentences[doc_id].update(_SENTENCE_END.split(match_form(text)))
    all_sentences = [s for group in sentences.values() for s in group]
    alias_forms = defaultdict(set)
    for name, (key, _) in resolution.items():
        alias_forms[merge.get(key, key)].add(match_form(name))

    def passes_r3(key: str) -> bool:
        entity = entities.get(key)
        if entity and any(_is_named(surface, texts.get((doc, idx), ""))
                          for doc, idx, surface in entity["mentions"]):
            return True
        forms = {key} | alias_forms[key]
        return sum(any(_mentions(f, s) for f in forms) for s in all_sentences) >= 2

    for key in sorted(candidates):
        if not passes_r3(key):
            rejected[key] = "neither named nor recurring (R3)"
    admitted = candidates - set(rejected)

    while True:                          # R1, until stable
        linked = {k for s, _, o in all_edges if s in admitted and o in admitted for k in (s, o)}
        stranded = admitted - linked
        if not stranded:
            break
        for key in stranded:
            rejected[key] = "no edge to an admitted entity (R1)"
        admitted -= stranded

    edges = {k: prov for k, prov in all_edges.items() if k[0] in admitted and k[2] in admitted}
    report = {
        "level": level,
        "ambiguous_acronyms": ambiguous,
        "self_loops_from_merges": self_loops,
        "merged_by_layer": Counter(layer for _, layer in resolution.values()),
    }
    return Graph(dict(entities), edges, admitted, rejected, report, candidates)


# --- layer 3: proposals only ----------------------------------------------------

async def propose_merges(graph: Graph, threshold: float = 0.90) -> list[dict]:
    """
    Pairs of entities whose names embed close together. PROPOSALS for a person
    to label; nothing here merges anything.

    A pair is proposed only if it also passes two guards:
      qualifier guard  neither name's words are a strict subset of the other's —
                       {shield} vs {shield, advanced} is two products, never one
      number guard     names with different numbers never merge — RFC 8417 vs 8935
    and both entities have the same type, and no one has labelled the pair already.

    Measured on this corpus: 14 of 16 proposals were right, and no threshold
    separates right from wrong — a wrong pair scored 0.946, above 11 right ones.
    Similarity finds candidates; it cannot decide them.
    """
    from app.rag.embed import embed_texts          # needs an API key; only this function does

    decided = set(load_decisions()["keep_apart"])
    keys = sorted(graph.candidates)
    names = [graph.entities[k]["canonical_name"] if k in graph.entities else k for k in keys]
    vectors = await embed_texts(names)
    similarity = vectors @ vectors.T

    proposals = []
    for i, j in zip(*np.where(np.triu(similarity, k=1) >= threshold)):
        a, b = keys[i], keys[j]
        if frozenset((a, b)) in decided:
            continue
        words_a, words_b = set(a.split()), set(b.split())
        if words_a < words_b or words_b < words_a:
            continue
        if re.findall(r"\d+", a) != re.findall(r"\d+", b):
            continue
        type_a = graph.entities.get(a, {}).get("type")
        type_b = graph.entities.get(b, {}).get("type")
        if type_a and type_b and type_a != type_b:
            continue
        proposals.append({"a": a, "b": b, "name_a": names[i], "name_b": names[j],
                          "type": type_a or type_b, "cosine": round(float(similarity[i, j]), 3)})
    return sorted(proposals, key=lambda p: -p["cosine"])


# --- persistence ------------------------------------------------------------

CHUNK_FIELDS = ("doc_id", "filename", "chunk_index", "page_start", "page_end", "ocr")
# What the graph needs from an extraction result. The per-run edge lists the
# pilot uses for its stability statistics are left out.
RESULT_FIELDS = ("entities", "relations", "dropped", "verb_splits", "usage")


def chunk_id(chunk: dict) -> str:
    """Stable across deletions: row positions in the vector store shift, (doc_id, chunk_index) does not."""
    return f"{chunk['doc_id']}:{chunk['chunk_index']}"


def record_for(chunk: dict, result: dict) -> dict:
    """One persisted record: the chunk's provenance plus what extraction found in it."""
    record = {field: chunk[field] for field in CHUNK_FIELDS}
    record |= {field: result[field] for field in RESULT_FIELDS if field in result}
    if "error" in result:
        record["error"] = result["error"]
    return record


class GraphStore:
    """
    Holds the extraction records (persisted) and the graph derived from them
    (never persisted). Every change to the records rebuilds the graph, so the
    graph can never disagree with them.
    """

    def __init__(self):
        self.records: dict[str, dict] = {}
        self.meta: dict = {}
        self.graph: Graph | None = None
        self._load()

    # --- public API -------------------------------------------------

    def add_document(self, doc_id: str, chunks: list[dict], results: list[dict]) -> int:
        """
        Store one document's extraction results and rebuild. Returns the number
        of chunk records added (a chunk whose extraction failed is kept with its
        error, so the failure stays visible).
        """
        from app.config import CHAT_MODEL, EXTRACT_MIN_AGREE, EXTRACT_RUNS, GRAPH_PROMPT_VERSION
        if not self.meta:
            self.meta = {"prompt_version": GRAPH_PROMPT_VERSION, "model": CHAT_MODEL,
                         "runs": EXTRACT_RUNS, "min_agree": EXTRACT_MIN_AGREE}
        if self.meta["prompt_version"] != GRAPH_PROMPT_VERSION:
            # A graph must come from one prompt: mixing versions would mix standards.
            raise ValueError(f"Graph holds prompt {self.meta['prompt_version']}; "
                             f"cannot add records from {GRAPH_PROMPT_VERSION}.")
        for chunk, result in zip(chunks, results):
            self.records[chunk_id(chunk)] = record_for(chunk, result)
        self._save()
        self._refresh()
        return len(chunks)

    def delete_document(self, doc_id: str) -> int:
        """Remove every record from one document and rebuild. Returns the number removed."""
        doomed = [cid for cid, record in self.records.items() if record["doc_id"] == doc_id]
        for cid in doomed:
            del self.records[cid]
        if doomed:
            self._save()
            self._refresh()
        return len(doomed)

    def texts(self) -> dict:
        from app.rag.store import store
        return {(c["doc_id"], c["chunk_index"]): c["text"] for c in store.metadata}

    def rebuild(self, level: str = "layer2", extra_merges: dict | None = None) -> Graph:
        return build_graph(self.records.values(), self.texts(), level, extra_merges=extra_merges)

    def is_empty(self) -> bool:
        return not self.records

    # --- internals --------------------------------------------------

    def _refresh(self) -> None:
        self.graph = self.rebuild() if self.records else None

    def _save(self) -> None:
        """Write to a temporary file, then rename: a crash mid-write never leaves half a file."""
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        temporary = EXTRACTIONS_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps({**self.meta, "chunks": self.records}))
        temporary.replace(EXTRACTIONS_PATH)

    def _load(self) -> None:
        if EXTRACTIONS_PATH.exists():
            data = json.loads(EXTRACTIONS_PATH.read_text())
            self.records = data.pop("chunks")
            self.meta = data
            self._refresh()


# Single shared instance, created once at import.
graph_store = GraphStore()
