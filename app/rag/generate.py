# app/rag/generate.py
from openai import AsyncOpenAI

from app.config import require_api_key, CHAT_MODEL

client = AsyncOpenAI(api_key=require_api_key())


class GenerationError(Exception):
    """Raised when the LLM call fails."""


SYSTEM_PROMPT = """You are a document assistant. Answer questions using ONLY the context passages provided.

Rules:
- Base your answer strictly on the numbered context passages.
- If the context does not contain the answer, say the uploaded documents do not cover it. Do not guess.
- Do not use knowledge from outside the context, even if you are confident about it.
- Cite the passages you used by number, like [1] or [2].
- Be concise. Do not repeat the question back."""

NO_CONTEXT_REPLY = (
    "I couldn't find anything in the uploaded documents that answers that. "
    "Try rephrasing, or upload a document covering this topic."
)


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered, labelled passages."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] (source: {chunk['filename']}, section {chunk['chunk_index']})\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


async def generate_answer(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> str:
    """
    Generate an answer grounded in the retrieved chunks.
    An empty `chunks` list short-circuits without calling the LLM.
    """
    if not chunks:
        return NO_CONTEXT_REPLY

    context = build_context(chunks)
    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

    messages = (history or []) + [{"role": "user", "content": user_message}]

    try:
        response = await client.responses.create(
            model=CHAT_MODEL,
            instructions=SYSTEM_PROMPT,
            input=messages,
        )
    except Exception as e:
        print(f"[LLM ERROR] {type(e).__name__}: {e}")
        raise GenerationError("The AI service is temporarily unavailable.")

    return response.output_text
