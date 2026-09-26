# app/routes/graph_chat.py
"""
Step 8 (EXPERIMENTAL): chat with graph passages beside search passages.

A separate endpoint, so /api/chat stays byte-identical. It answers 403 unless
GRAPH_RAG_ENABLED is set, and keeps its own session history, so one mode's answers
never leak into the other's rewrites. Not evidence that graph RAG helps: Step 7
found it does not on this corpus.
"""
from fastapi import APIRouter, HTTPException

from app.config import GRAPH_RAG_ENABLED
from app.rag.generate import GenerationError, fit_context, rewrite_question
from app.rag.graph_answer import generate_graph_answer
from app.rag.graph_retrieve import graph_evidence
from app.rag.retrieve import retrieve
from app.rag.store import store
from app.routes.chat import MAX_HISTORY, SNIPPET_LENGTH
from app.schemas import ChatRequest, GraphChatResponse, GraphEvidence, GraphSource

router = APIRouter(prefix="/api", tags=["chat: experimental graph mode"])

GRAPH_SESSIONS: dict[str, list[dict]] = {}
DISABLED = "Experimental graph mode is off. Set GRAPH_RAG_ENABLED=true in .env and restart to use it."


def as_source(chunk: dict) -> GraphSource:
    return GraphSource(
        doc_id=chunk["doc_id"],
        filename=chunk["filename"],
        chunk_index=chunk["chunk_index"],
        score=round(chunk["score"], 3),
        snippet=chunk["text"][:SNIPPET_LENGTH].strip(),
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_end"),
        ocr=chunk.get("ocr", False),
        retrieval=chunk.get("retrieval", "vector"),
        graph=GraphEvidence(**chunk["graph"]) if chunk.get("graph") else None,
    )


@router.post("/chat/graph", response_model=GraphChatResponse)
async def chat_graph(payload: ChatRequest):
    """
    EXPERIMENTAL. Answers from search passages plus up to two graph passages, each
    carrying the path and quote that reached it. Step 7 found graph retrieval does
    not improve retrieval on this corpus; compare with POST /api/chat.
    """
    if not GRAPH_RAG_ENABLED:
        raise HTTPException(status_code=403, detail=DISABLED)
    if store.is_empty():
        raise HTTPException(status_code=400, detail="No documents have been uploaded yet.")

    history = GRAPH_SESSIONS.get(payload.session_id, [])
    search_query = await rewrite_question(payload.question, history)

    # 1. Retrieve: today's vector retrieval, then graph passages beside it (none if it found nothing)
    try:
        vector_chunks = fit_context(await retrieve(search_query))    # the same budget as /api/chat
        graph_chunks = await graph_evidence(search_query, vector_chunks)
    except Exception as e:
        print(f"[GRAPH RETRIEVAL ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail="Could not search the documents.")

    # 2. Generate
    try:
        answer, graph_used = await generate_graph_answer(payload.question, vector_chunks, graph_chunks, history)
    except GenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # 3. This mode's own history
    GRAPH_SESSIONS[payload.session_id] = (history + [
        {"role": "user", "content": payload.question},
        {"role": "assistant", "content": answer},
    ])[-MAX_HISTORY:]

    # 4. Sources in prompt order, so [n] in the answer is sources[n-1]
    return GraphChatResponse(
        answer=answer,
        sources=[as_source(c) for c in vector_chunks + graph_used],
        session_id=payload.session_id,
        search_query=search_query,
    )


@router.get("/chat/graph/status")
async def graph_status():
    """Whether this server has graph mode on, so the UI can say so instead of failing."""
    return {"enabled": GRAPH_RAG_ENABLED, "detail": None if GRAPH_RAG_ENABLED else DISABLED}


@router.post("/chat/graph/reset")
async def reset_graph_chat(session_id: str = "default"):
    GRAPH_SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id, "mode": "vector+graph"}
