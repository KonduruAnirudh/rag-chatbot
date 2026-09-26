# eval/step8_demo.py
"""
Step 8 demonstration: real answers from vector-only mode and graph mode, side by side.

A DEMONSTRATION, NOT EVIDENCE. Step 7 measured whether graph retrieval improves
retrieval on this corpus, and it does not. The graph passages it adds are gold
about 5% of the time (2 of 24 on the development half, 1 of 29 held out). This
shows what the experimental path does to real answers. It never loads a held-out
question: it uses the development half and the drafts that were not kept (never
scored by any arm), plus the standing refusal question.

Both endpoints answer every question, with real model calls. It reports:
  1. refusal flips, both ways - vector-only declined and graph mode answered, and
     the reverse - with on-topic questions (a possible gain) and off-topic questions
     naming a corpus entity (a failure) counted separately: one mechanism, two results
  2. citations: how often graph-mode answers cite a graph passage, and whether the
     cited passages are gold - next to how often search passages are cited
  3. latency, p50 and p95: the whole request in each mode, and the graph step split
     into the second embedding call and the traversal itself

Whether an answer declined is read from its text: the fixed refusal, or a "not
covered" phrase with no citation ("partial" when it also cites passages). That
reading is a first pass. The labelling file lists every flip, every cited graph
passage and Q38 for a person to judge against the documents.

Saves eval/step8_runs/demo-<time>.json and demo-<time>-for-labelling.txt.
About 80 answers: a few cents.

Run from the project root:
    python -m eval.step8_demo
"""
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from app.main import app
from app.rag import graph_retrieve as G
from app.rag.generate import NO_CONTEXT_REPLY
from app.rag.store import store
from app.routes import graph_chat
from eval.check_graph_questions import DOCUMENTS

QUESTIONS = Path("eval/graph_questions.yaml")
RUNS = Path("eval/step8_runs")
STANDING_REFUSAL = {"id": "FRANCE", "question": "What is the capital of France?", "category": "off-topic",
                    "off_topic_kind": "names-nothing", "gold_chunks": [], "answer_facts": [], "keep": None}
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
DECLINE = re.compile(
    r"\b(?:do(?:es)?\s+not|don't|doesn't)\s+(?:cover|contain|include|mention|specify|state|say|describe|"
    r"provide|address|explain|discuss)\b|\bnot\s+(?:covered|mentioned|specified|stated|described|addressed|"
    r"included|provided)\b|\bno\s+(?:information|mention|details?)\b|\bcould(?:n't| not)\s+find\b", re.I)
MODES = {"vector": "/api/chat", "graph": "/api/chat/graph"}


def label(chunk: dict) -> str:
    prefix = {filename: p for p, filename in DOCUMENTS.items()}
    return f"{prefix.get(chunk['filename'], chunk['filename'])}:{chunk['chunk_index']}"


def cited(answer: str, n_sources: int) -> list[int]:
    numbers = [int(x) for group in CITATION.findall(answer) for x in group.split(",")]
    return sorted({n for n in numbers if 1 <= n <= n_sources})


def reading(answer: str, n_sources: int) -> str:
    if answer.strip() == NO_CONTEXT_REPLY:
        return "refused"                               # structural: no model call
    if DECLINE.search(answer):
        return "partial" if cited(answer, n_sources) else "declined"
    return "answered"


def declined(r: str) -> bool:
    return r in ("refused", "declined")


def group_of(q: dict) -> str:
    if q["id"] == "FRANCE":
        return "standing refusal"
    return f"off-topic, {q['off_topic_kind']}" if q["category"] == "off-topic" else "on-topic"


def pct(values, share):
    ordered = sorted(values)
    return ordered[max(0, round(share * len(ordered)) - 1)] if ordered else float("nan")


def run(questions: list[dict]) -> list[dict]:
    timings = []                                      # per graph-mode request: (whole graph step, embedding)
    real_evidence, real_embed = graph_chat.graph_evidence, G.embed_query
    embed_ms = []

    async def timed_embed(text):
        started = time.perf_counter()
        vector = await real_embed(text)
        embed_ms.append((time.perf_counter() - started) * 1000)
        return vector

    async def timed_evidence(question, vector_chunks):
        before, started = len(embed_ms), time.perf_counter()
        evidence = await real_evidence(question, vector_chunks)
        timings.append(((time.perf_counter() - started) * 1000, sum(embed_ms[before:])))
        return evidence

    graph_chat.graph_evidence, G.embed_query = timed_evidence, timed_embed
    real_flag, graph_chat.GRAPH_RAG_ENABLED = graph_chat.GRAPH_RAG_ENABLED, True
    results = []
    try:
        with TestClient(app) as tc:
            for q in questions:
                record = {k: q[k] for k in ("id", "question", "category", "off_topic_kind", "gold_chunks", "answer_facts")}
                record["group"] = group_of(q)
                for mode, path in MODES.items():
                    graph_calls = len(timings)
                    started = time.perf_counter()
                    response = tc.post(path, json={"question": q["question"], "session_id": f"demo-{mode}-{q['id']}"})
                    elapsed = (time.perf_counter() - started) * 1000
                    body = response.json()
                    sources = body.get("sources", [])
                    numbers = cited(body.get("answer", ""), len(sources))
                    record[mode] = {
                        "status": response.status_code,
                        "answer": body.get("answer", body.get("detail")),
                        "reading": reading(body.get("answer", ""), len(sources)),
                        "request_ms": elapsed,
                        "graph_ms": timings[graph_calls] if len(timings) > graph_calls else None,
                        "sources": [{
                            "n": i + 1, "chunk": label(s), "retrieval": s.get("retrieval", "vector"),
                            "gold": label(s) in q["gold_chunks"], "cited": i + 1 in numbers,
                            "links": [{k: l[k] for k in ("reached_from", "to", "relation", "quote")}
                                      for l in (s.get("graph") or {}).get("links", [])],
                        } for i, s in enumerate(sources)],
                    }
                results.append(record)
                v, g = record["vector"]["reading"], record["graph"]["reading"]
                print(f"  {q['id']:<7} {record['group']:<26} vector {v:<9} graph {g:<9}"
                      f" graph passages cited {sum(s['cited'] for s in record['graph']['sources'] if s['retrieval'] == 'graph')}"
                      f"/{sum(s['retrieval'] == 'graph' for s in record['graph']['sources'])}")
    finally:
        graph_chat.graph_evidence, G.embed_query = real_evidence, real_embed
        graph_chat.GRAPH_RAG_ENABLED = real_flag
    return results


def summarise(results: list[dict]) -> dict:
    summary = {"flips": {}, "citations": {}, "latency": {}}
    print("\n1. REFUSAL FLIPS   (declined = the fixed refusal, or 'not covered' with no citation)")
    print(f"  {'group':<27}{'questions':>10}{'both answer':>13}{'both decline':>14}"
          f"{'vector declined, graph answered':>33}{'vector answered, graph declined':>33}")
    for group in ("on-topic", "off-topic, names-entity", "off-topic, names-nothing", "standing refusal"):
        rows = [r for r in results if r["group"] == group]
        flip_on = [r["id"] for r in rows if declined(r["vector"]["reading"]) and not declined(r["graph"]["reading"])]
        flip_off = [r["id"] for r in rows if not declined(r["vector"]["reading"]) and declined(r["graph"]["reading"])]
        both_decline = sum(declined(r["vector"]["reading"]) and declined(r["graph"]["reading"]) for r in rows)
        summary["flips"][group] = {"questions": len(rows), "vector_declined_graph_answered": flip_on,
                                   "vector_answered_graph_declined": flip_off, "both_decline": both_decline}
        print(f"  {group:<27}{len(rows):>10}{len(rows) - both_decline - len(flip_on) - len(flip_off):>13}"
              f"{both_decline:>14}{len(flip_on):>7} {str(flip_on):<25}{len(flip_off):>7} {str(flip_off)}")

    print("\n2. CITATIONS IN GRAPH MODE")
    graph_answers = [r for r in results if any(s["retrieval"] == "graph" for s in r["graph"]["sources"])]
    shown = [(r, s) for r in graph_answers for s in r["graph"]["sources"] if s["retrieval"] == "graph"]
    search = [(r, s) for r in graph_answers for s in r["graph"]["sources"] if s["retrieval"] == "vector"]
    cited_graph = [(r, s) for r, s in shown if s["cited"]]
    on_topic = lambda pairs: [(r, s) for r, s in pairs if r["group"] == "on-topic"]
    c = summary["citations"] = {
        "answers_with_graph_passages": len(graph_answers),
        "answers_citing_a_graph_passage": sorted({r["id"] for r, _ in cited_graph}),
        "graph_passages_shown": len(shown), "graph_passages_cited": len(cited_graph),
        "search_passages_shown": len(search), "search_passages_cited": sum(s["cited"] for _, s in search),
        "on_topic_graph_shown": len(on_topic(shown)), "on_topic_graph_gold": sum(s["gold"] for _, s in on_topic(shown)),
        "cited_and_gold": [(r["id"], s["chunk"]) for r, s in cited_graph if s["gold"]],
        "cited_not_gold_on_topic": [(r["id"], s["chunk"]) for r, s in on_topic(cited_graph) if not s["gold"]],
        "cited_off_topic": [(r["id"], s["chunk"]) for r, s in cited_graph if r["group"] != "on-topic"],
    }
    rate = lambda a, b: f"{a}/{b} = {a / b:.0%}" if b else "0/0"
    print(f"  answers given at least one graph passage:        {c['answers_with_graph_passages']}")
    print(f"  answers citing a graph passage:                   {rate(len(c['answers_citing_a_graph_passage']), c['answers_with_graph_passages'])}"
          f"   {c['answers_citing_a_graph_passage']}")
    print(f"  graph passages cited, of those shown:             {rate(c['graph_passages_cited'], c['graph_passages_shown'])}")
    print(f"  search passages cited, of those shown (same answers): {rate(c['search_passages_cited'], c['search_passages_shown'])}")
    print(f"  gold rate of graph passages shown (on-topic):     {rate(c['on_topic_graph_gold'], c['on_topic_graph_shown'])}")
    print(f"  cited and gold:                                   {len(c['cited_and_gold'])}  {c['cited_and_gold']}")
    print(f"  cited and not gold (on-topic):                    {len(c['cited_not_gold_on_topic'])}  {c['cited_not_gold_on_topic']}")
    print(f"  cited in an off-topic answer:                     {len(c['cited_off_topic'])}  {c['cited_off_topic']}")

    print("\n3. LATENCY (ms)")
    lat = summary["latency"]
    for mode in MODES:
        values = [r[mode]["request_ms"] for r in results if r[mode]["reading"] != "refused"]
        lat[f"{mode}_request"] = (pct(values, .5), pct(values, .95))
        print(f"  whole request, {mode:<6} mode (answered questions): p50 {lat[f'{mode}_request'][0]:6.0f}   p95 {lat[f'{mode}_request'][1]:6.0f}")
    steps = [r["graph"]["graph_ms"] for r in results if r["graph"]["graph_ms"] and r["graph"]["graph_ms"][1] > 0]
    for name, values in (("graph step, whole", [a for a, _ in steps]), ("  2nd embedding call", [b for _, b in steps]),
                         ("  traversal itself", [a - b for a, b in steps])):
        lat[name.strip()] = (pct(values, .5), pct(values, .95))
        print(f"  {name:<30} p50 {lat[name.strip()][0]:7.1f}   p95 {lat[name.strip()][1]:7.1f}   ({len(values)} requests)")
    return summary


def labelling_file(results: list[dict], path: Path) -> None:
    """Every flip, every answer citing a graph passage, and Q38 - for a person to judge against the documents."""
    text = {label(c): c["text"] for c in store.metadata}
    chosen = [r for r in results
              if r["id"] == "Q38"
              or declined(r["vector"]["reading"]) != declined(r["graph"]["reading"])
              or any(s["cited"] and s["retrieval"] == "graph" for s in r["graph"]["sources"])]
    lines = ["STEP 8 DEMONSTRATION - for labelling. Not evidence: see eval/step8_demo.py.",
             "For each: is each answer correct against the documents (correct / partly / wrong / declined)?",
             "For each cited graph passage: does the text of that passage support the sentence that cites it?", ""]
    for r in chosen:
        lines += ["=" * 90, f"{r['id']} [{r['group']}] {r['question']}",
                  f"  gold chunks: {r['gold_chunks'] or '-'}", f"  answer facts: {r['answer_facts'] or '-'}"]
        for mode in MODES:
            m = r[mode]
            lines += [f"\n  {mode.upper()} ({m['reading']}): {m['answer']}",
                      "  sources: " + ", ".join(f"[{s['n']}] {s['chunk']}{' GRAPH' if s['retrieval'] == 'graph' else ''}"
                                                f"{' gold' if s['gold'] else ''}{' CITED' if s['cited'] else ''}"
                                                for s in m["sources"])]
        for s in r["graph"]["sources"]:
            if s["retrieval"] == "graph":
                lines += [f"\n  graph passage [{s['n']}] {s['chunk']}{' (gold)' if s['gold'] else ''}"
                          f"{' - CITED' if s['cited'] else ' - not cited'}"]
                lines += [f"    path: question -> {l['reached_from']} -> {l['to']}   quote: {l['quote']}" for l in s["links"]]
                lines += ["    text: " + " ".join(text[s["chunk"]].split())]
        lines += ["", "  LABEL vector: ______   graph: ______   cited graph passage supports its sentence: ______", ""]
    path.write_text("\n".join(lines))


def main():
    loaded = [q for q in yaml.safe_load(QUESTIONS.read_text()) if q["split"] != "heldout"]
    assert not any(q["split"] == "heldout" for q in loaded)
    questions = loaded + [STANDING_REFUSAL]
    print(f"STEP 8 DEMONSTRATION - not evidence. {len(questions)} questions (no held-out question), both modes\n")
    results = run(questions)
    summary = summarise(results)

    RUNS.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    path = RUNS / f"demo-{stamp}.json"
    path.write_text(json.dumps({"summary": summary, "questions": results}, indent=1))
    labelling_file(results, RUNS / f"demo-{stamp}-for-labelling.txt")
    print(f"\nsaved {path} and {RUNS / f'demo-{stamp}-for-labelling.txt'}")


if __name__ == "__main__":
    main()
