# eval/graph_precision_sample.py
"""
Step 6b: a blind random sample of the graph's edges, for measuring precision
of the graph that Step 7 will actually traverse (after resolution, the labelled
layer-3 merges, and admission).

Writes:
  eval/graph_precision_sample_for_labelling.txt  the edges, their quotes and passages;
                                                  no votes, no admission status
  eval/pilot_runs/graph_precision_sample.json    everything hidden from the labeller,
                                                  for scoring afterwards

The seed is fixed, so rerunning draws the same 40 edges from the same graph.

Run from the project root:
    python -m eval.graph_precision_sample
"""
import json
import random
import textwrap
from pathlib import Path

from app.rag.graph import graph_store
from app.rag.graph_extract import EXTRACTION_PROMPT, clean

SEED = 20260925
SAMPLE_SIZE = 40
TO_LABEL = Path("eval/graph_precision_sample_for_labelling.txt")
HIDDEN = Path("eval/pilot_runs/graph_precision_sample.json")

# The 7 chunks the extraction prompt was tuned on; recorded (hidden) so precision
# can be split into tuned and untuned text afterwards.
TUNED = {("AWS-WhitePaper-Sample.pdf", i) for i in (39, 153, 154, 156, 157, 158)} | {("rfc8935.pdf", 15)}

VERB_DEFINITIONS = EXTRACTION_PROMPT.split("RELATIONSHIPS")[1].split("RULES")[0].strip()

HEADER = f"""GRAPH PRECISION SAMPLE - {SAMPLE_SIZE} edges drawn at random (seed {SEED}) from the graph
Step 7 will traverse. Label from THIS file. List the wrong ones.

LABELLING STANDARD (v5, fixed). An edge is WRONG if:
  1. The verb doesn't match its definition (below).
  2. The direction is reversed.
  3. The quote doesn't STATE the relation - a purpose, benefit or effect is inferred.
     "X for Y" and "access to Y" do not state that X contains Y.
  4. An endpoint isn't the real participant in the quote (a substitute, or the wrong actor).
An edge stated in more than one chunk shows every quote. It is CORRECT if at least one
quote states it: the graph shows all of its quotes as evidence.

Names are the graph's canonical names; "as written" shows the quote's own wording
where it differs.

VERBS {VERB_DEFINITIONS}
"""


def main():
    graph = graph_store.graph
    texts = graph_store.texts()
    edges = sorted(graph.edges)
    sample = random.Random(SEED).sample(edges, SAMPLE_SIZE)

    lines, hidden = [HEADER], []
    for number, (subject, verb, obj) in enumerate(sample, start=1):
        eid = f"E{number}"
        name = lambda key: graph.entities[key]["canonical_name"]
        lines += ["=" * 78, f"{eid:<4} {name(subject)}  --{verb}-->  {name(obj)}"]
        provenance = graph.edges[(subject, verb, obj)]
        for p in provenance:
            written = f"{p['subject_as_written']} ... {p['object_as_written']}"
            if written != f"{name(subject)} ... {name(obj)}":
                lines.append(f"     as written: {written}")
            lines.append(f"     source: {p['filename']}, pages {p['page_start']}-{p['page_end']}, "
                         f"chunk {p['chunk_index']}{', OCR' if p['ocr'] else ''}")
            lines.append(textwrap.fill(f'quote: "{p["evidence"]}"', width=78, break_on_hyphens=False,
                                       initial_indent="     ", subsequent_indent="            "))
            passage = clean(texts.get((p["doc_id"], p["chunk_index"]), ""))
            lines.append("     passage:")
            lines.append(textwrap.indent(passage, "       | "))
        lines.append("")
        hidden.append({
            "id": eid, "subject": subject, "relation": verb, "object": obj,
            "quotes": len(provenance),
            "votes": [p["votes"] for p in provenance],
            "admitted": True,                                    # sampled from admitted edges only
            "subject_neighbours": graph.neighbour_count(subject),
            "object_neighbours": graph.neighbour_count(obj),
            "tuned_chunk": any((p["filename"], p["chunk_index"]) in TUNED for p in provenance),
            "sources": [f"{p['filename']}:{p['chunk_index']}" for p in provenance],
        })

    TO_LABEL.write_text("\n".join(lines))
    HIDDEN.parent.mkdir(parents=True, exist_ok=True)
    HIDDEN.write_text(json.dumps({"seed": SEED, "graph_edges": len(edges), "prompt_version": graph_store.meta.get("prompt_version"),
                                  "sample": hidden}, indent=2))
    multi = sum(1 for h in hidden if h["quotes"] > 1)
    print(f"Sampled {SAMPLE_SIZE} of {len(edges)} edges (seed {SEED}); {multi} have more than one quote.")
    print(f"To label: {TO_LABEL}  (no votes or admission status)")
    print(f"Hidden details for scoring: {HIDDEN}")


if __name__ == "__main__":
    main()
