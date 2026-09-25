# eval/linking_recall.py
"""
Step 7, step zero: can a question be linked to the graph entities it is about?

If the entities a question is about can't be found, no traversal policy can
help — so this runs before any retrieval arm is built.

Reads the question set labelled by a person (keep: yes, and `entities`: what
each question is about, in plain words). Each intended entity ends up as one of:

  linked        the linker found it in the question
  missed        it is a graph node, but the linker did not find it in the question
  not admitted  extraction produced it, but admission rejected it (the reason is shown)
  not in graph  extraction never produced it, under any name

The linker is deliberately simple and deterministic: the names of every admitted
entity (its key, every surface form, acronyms merged into it) are matched as whole
words with an optional plural, longest match first, so "Shield Advanced finding"
wins over "Shield Advanced". No model calls.

Run from the project root:
    python -m eval.linking_recall                    # eval/graph_questions.yaml
    python -m eval.linking_recall path/to/questions.yaml
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from app.rag.graph import graph_store, is_generic, load_decisions, normalise
from app.rag.graph_extract import match_form

QUESTIONS = Path("eval/graph_questions.yaml")
HUB = 10    # neighbours; the report counts how many extra links are hubs


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


def key_for(name: str, graph) -> str | None:
    """The graph key a person's plain-words entity name refers to, if any."""
    merges = load_decisions()["merges"]
    candidates = [match_form(name), normalise(name)[0]]
    candidates += [c[:-1] for c in candidates if c.endswith("s")]
    for candidate in candidates:
        candidate = merges.get(candidate, candidate)
        if candidate in graph.entities:
            return candidate
    return None


def classify(name: str, linked: list[str], graph) -> tuple[str, str]:
    key = key_for(name, graph)
    if key is None:
        # Longer names that contain its words: admitted nodes, and extracted names admission rejected
        words = set(match_form(name).split())
        near = sorted(k for k in graph.entities if words and words <= set(k.split()))
        shown = [k if k in graph.admitted else f"{k} (rejected: {graph.rejected.get(k, 'not admitted')})"
                 for k in near[:4]]
        return "not in graph", f"longer names containing its words: {shown or 'none'}"
    if key not in graph.admitted:
        return "not admitted", graph.rejected.get(key, "not admitted")
    return ("linked", key) if key in linked else ("missed", key)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else QUESTIONS
    if not path.exists():
        raise SystemExit(f"{path} not found - label the draft (eval/graph_questions_draft.yaml) "
                         f"and save it as {QUESTIONS}.")
    questions = [q for q in yaml.safe_load(path.read_text()) if q.get("keep") is True]
    graph = graph_store.graph
    forms = entity_forms(graph)

    outcomes = defaultdict(Counter)          # category -> outcome -> count
    fully_linked = Counter()
    per_category = Counter()
    extra_links, extra_hubs, extra_generic = Counter(), Counter(), Counter()

    print(f"{len(questions)} questions from {path}; {len(graph.admitted)} graph entities, {len(forms)} names\n")
    for q in questions:
        category = q.get("category") or "unlabelled"
        per_category[category] += 1
        linked = link(q["question"], forms)
        intended = q.get("entities") or []
        results = [(name, *classify(name, linked, graph)) for name in intended]
        intended_keys = {detail for _, outcome, detail in results if outcome in ("linked", "missed")}
        extras = [k for k in linked if k not in intended_keys]

        for _, outcome, _ in results:
            outcomes[category][outcome] += 1
        if intended and all(outcome == "linked" for _, outcome, _ in results):
            fully_linked[category] += 1
        extra_links[category] += len(extras)
        extra_hubs[category] += sum(graph.neighbour_count(k) >= HUB for k in extras)
        extra_generic[category] += sum(is_generic(k) for k in extras)

        print(f"{q['id']} [{category}] {q['question']}")
        for name, outcome, detail in results:
            print(f"     {outcome:<13} {name!r:<34} {detail}")
        print(f"     linker found: {[(k, graph.neighbour_count(k)) for k in linked] or 'nothing'}"
              + (f"   extra: {extras}" if extras else ""))
        print()

    print("=" * 78)
    print("LINKING RECALL  (intended entities, by outcome)")
    print("-" * 78)
    print(f"  {'category':<14}{'questions':>10}{'linked':>9}{'missed':>9}{'not adm.':>10}{'not in g.':>11}"
          f"{'all linked':>12}{'extra links':>13}")
    total = Counter()
    for category in sorted(per_category):
        o = outcomes[category]
        total.update(o)
        print(f"  {category:<14}{per_category[category]:>10}{o['linked']:>9}{o['missed']:>9}"
              f"{o['not admitted']:>10}{o['not in graph']:>11}{fully_linked[category]:>12}"
              f"{extra_links[category]:>8} ({extra_hubs[category]} hubs)")
    intended_total = sum(total.values())
    if intended_total:
        print(f"\n  linking recall: {total['linked']}/{intended_total} = {total['linked'] / intended_total:.0%}"
              f"   (missed {total['missed']}, not admitted {total['not admitted']}, "
              f"not in graph {total['not in graph']})")


if __name__ == "__main__":
    main()
