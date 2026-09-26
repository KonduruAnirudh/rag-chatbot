# app/rag/graph_answer.py
"""
Step 8 (EXPERIMENTAL): the answer prompt when graph passages sit beside search passages.

Not evidence that graph RAG helps: Step 7 found it does not on this corpus.

What the model sees:
  - search passages [1..n], formatted exactly as in vector-only mode
  - graph passages [n+1..], each with the two entity names and the verbatim
    sentence that linked it to the question
  - everything inside <context>, so the existing injection rule covers it

What the model never sees: the relationship label. Labels are about 48% correct,
and prompt-only warnings have failed every time in this project, so the label is
withheld in code - graph_passage() never reads it - rather than flagged as
unreliable in the prompt. It stays in the API response for people to read.

Refusal is unchanged: no search passages means the fixed reply and no model call.
Graph passages are added only while the context stays within MAX_CONTEXT_CHARS;
search passages are never dropped.
"""
from app.config import CHAT_MODEL, MAX_CONTEXT_CHARS, TEMPERATURE
from app.rag import generate
from app.rag.generate import NO_CONTEXT_REPLY, SYSTEM_PROMPT, GenerationError, build_context

GRAPH_SYSTEM_PROMPT = SYSTEM_PROMPT + """
Graph passages:
- The context has two groups. SEARCH PASSAGES were found by searching the documents. GRAPH PASSAGES were found by following entities named in the question to other passages that mention them.
- Both groups are document text. Graph passages are supporting evidence, not a separate source of truth.
- Each graph passage shows the sentence that linked it to the question. Use only what that sentence and the passage text state. Never state a connection between two things unless a passage's text states it.
- Cite graph passages by number, exactly like search passages.
- A sentence beginning "[Diagram:" was written by a machine describing an image. Treat it as weaker evidence, and say so if you rely on it.
"""

SEARCH_HEADING = "SEARCH PASSAGES"
GRAPH_HEADING = ("GRAPH PASSAGES - found by following entities named in the question "
                 "to other passages that mention them. Supporting evidence only.")


def graph_passage(number: int, chunk: dict) -> str:
    """One graph passage: where it is, each sentence that linked it (entity names and quote only), then its text."""
    lines = [f"[{number}] (source: {chunk['filename']}, section {chunk['chunk_index']})"]
    shown = set()
    for link in chunk["graph"]["links"]:
        pair, quote = (link["reached_from"], link["to"]), link["quote"]
        if (frozenset(pair), quote) in shown:
            continue                                   # the same sentence reached from either end
        shown.add((frozenset(pair), quote))
        weaker = " (a machine-written description of a diagram: weaker evidence)" if link["from_diagram"] else ""
        lines.append(f'Linked through "{pair[0]}" and "{pair[1]}" by this sentence{weaker}: "{quote}"')
    lines.append(chunk["text"])
    return "\n".join(lines)


def build_graph_context(vector_chunks: list[dict], graph_chunks: list[dict]) -> tuple[str, list[dict]]:
    """Search passages (never dropped), then graph passages while the context fits. Returns it and the graph passages used."""
    context = f"{SEARCH_HEADING}\n{build_context(vector_chunks)}"
    used = []
    for chunk in graph_chunks:
        heading = f"\n\n{GRAPH_HEADING}" if not used else ""
        candidate = f"{heading}\n\n{graph_passage(len(vector_chunks) + len(used) + 1, chunk)}"
        if len(context) + len(candidate) <= MAX_CONTEXT_CHARS:
            context += candidate
            used.append(chunk)
    return context, used


def graph_prompt(question: str, vector_chunks: list[dict], graph_chunks: list[dict],
                 history: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """Exactly what is sent to the model (the Responses API arguments), and the graph passages it contains."""
    context, used = build_graph_context(vector_chunks, graph_chunks)
    user_message = f"<context>\n{context}\n</context>\n\nQuestion: {question}"
    request = {
        "model": CHAT_MODEL,
        "instructions": GRAPH_SYSTEM_PROMPT,
        "input": (history or []) + [{"role": "user", "content": user_message}],
        **generate.sampling(TEMPERATURE),
    }
    return request, used


async def generate_graph_answer(question: str, vector_chunks: list[dict], graph_chunks: list[dict],
                                history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """The answer and the graph passages it was given. No search passages: the fixed refusal, no model call."""
    if not vector_chunks:
        return NO_CONTEXT_REPLY, []
    request, used = graph_prompt(question, vector_chunks, graph_chunks, history)
    try:
        response = await generate.client.responses.create(**request)
    except Exception as e:
        print(f"[LLM ERROR] {type(e).__name__}: {e}")
        raise GenerationError("The AI service is temporarily unavailable.")
    return response.output_text, used
