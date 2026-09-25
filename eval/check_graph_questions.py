# eval/check_graph_questions.py
"""
Step 7: check the labelled question set before anything is measured with it.

  1. every question has all the required fields, with allowed values
  2. every gold chunk exists in the store
  3. every answer fact occurs verbatim in one of its question's gold chunks

"Verbatim" uses the same comparison as the extraction quote check (match_form):
case, whitespace and line-wrapped hyphens are ignored, nothing else. A fact that
only matches across the join of two chunks fails - it is not in either chunk.

Run from the project root after any edit to the labels; exits 1 on any problem:
    python -m eval.check_graph_questions                    # eval/graph_questions.yaml
    python -m eval.check_graph_questions path/to/questions.yaml
"""
import sys
from collections import Counter
from pathlib import Path

import yaml

from app.rag.graph_extract import match_form
from app.rag.store import store

QUESTIONS = Path("eval/graph_questions.yaml")
DOCUMENTS = {"AWS": "AWS-WhitePaper-Sample.pdf", "RFC": "rfc8935.pdf"}   # gold chunk prefix -> file

# field -> test its value must pass
REQUIRED = {
    "id":             lambda v: isinstance(v, str) and v != "",
    "question":       lambda v: isinstance(v, str) and v != "",
    "draft_category": lambda v: isinstance(v, str),
    "found_in":       lambda v: isinstance(v, str),
    "keep":           lambda v: isinstance(v, bool),
    "split":          lambda v: v in ("dev", "heldout", None),
    "category":       lambda v: v in ("plain", "relationship", "multi-hop", "off-topic"),
    "off_topic_kind": lambda v: v in ("names-nothing", "names-entity", None),
    "needs_graph":    lambda v: isinstance(v, bool),
    "entities":       lambda v: isinstance(v, list),
    "gold_chunks":    lambda v: isinstance(v, list),
    "answer_facts":   lambda v: isinstance(v, list),
    "notes":          lambda v: v is None or isinstance(v, str),
}


def chunk_texts() -> dict[str, str]:
    """Gold chunk id ("AWS:154") -> the chunk's text in match form."""
    prefix = {filename: p for p, filename in DOCUMENTS.items()}
    return {f"{prefix[c['filename']]}:{c['chunk_index']}": match_form(c["text"])
            for c in store.metadata if c["filename"] in prefix}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else QUESTIONS
    questions = yaml.safe_load(path.read_text())
    texts = chunk_texts()
    problems = []

    ids = Counter(q.get("id") for q in questions)
    problems += [f"{qid}: id used {n} times" for qid, n in ids.items() if n > 1]

    facts_checked = gold_checked = 0
    for q in questions:
        qid = q.get("id", "?")
        for field, allowed in REQUIRED.items():
            if field not in q:
                problems.append(f"{qid}: missing field '{field}'")
            elif not allowed(q[field]):
                problems.append(f"{qid}: '{field}' has a value that is not allowed: {q[field]!r}")

        gold = q.get("gold_chunks") or []
        for chunk in gold:
            gold_checked += 1
            if chunk not in texts:
                problems.append(f"{qid}: gold chunk {chunk} does not exist")
        for fact in q.get("answer_facts") or []:
            facts_checked += 1
            if not any(match_form(fact) in texts.get(chunk, "") for chunk in gold):
                problems.append(f"{qid}: answer fact not verbatim in any of its gold chunks {gold}: {fact!r}")

    kept = [q for q in questions if q.get("keep") is True]
    print(f"{path}: {len(questions)} questions, {len(kept)} kept; "
          f"checked {len(REQUIRED)} fields each, {gold_checked} gold chunks, {facts_checked} answer facts")
    print("kept by category:", dict(Counter(q.get("category") for q in kept)))
    print("kept by split:   ", dict(Counter(q.get("split") for q in kept)))
    for problem in problems:
        print("  PROBLEM", problem)
    print(f"\n{len(problems)} problems." if problems else "\nAll checks passed.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
