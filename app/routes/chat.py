# app/routes/chat.py
from fastapi import APIRouter, HTTPException

from app.schemas import ChatRequest, ChatResponse, Source
from app.rag.retrieve import retrieve
from app.rag.generate import generate_answer, rewrite_question, GenerationError
from app.rag.generate import fit_context
from app.rag.store import store

router = APIRouter(prefix="/api", tags=["chat"])

SESSIONS: dict[str, list[dict]] = {}
MAX_HISTORY = 6          # last 3 exchanges
SNIPPET_LENGTH = 200


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    if store.is_empty():
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet.",
        )

    history = SESSIONS.get(payload.session_id, [])

    # Resolve references before retrieval — "tell me more about that"
    # has no topic to match on until it's rewritten.
    search_query = await rewrite_question(payload.question, history)

    # 1. Retrieve
    try:
        chunks = await retrieve(search_query)
    except Exception as e:
        print(f"[RETRIEVAL ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail="Could not search the documents.")

    # The context budget: the prompt and the sources below drop the same passages
    chunks = fit_context(chunks)

    # 2. Generate
    try:
        answer = await generate_answer(payload.question, chunks, history)
    except GenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # 3. Update history (store the plain question, not the context-stuffed prompt)
    SESSIONS[payload.session_id] = (history + [
        {"role": "user", "content": payload.question},
        {"role": "assistant", "content": answer},
    ])[-MAX_HISTORY:]

    # 4. Shape the sources
    sources = [
        Source(
            doc_id=c["doc_id"],
            filename=c["filename"],
            chunk_index=c["chunk_index"],
            score=round(c["score"], 3),
            snippet=c["text"][:SNIPPET_LENGTH].strip(),
            page_start=c.get("page_start"),
            page_end=c.get("page_end"),
            ocr=c.get("ocr", False),
        )
        for c in chunks
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=payload.session_id,
        search_query=search_query,
    )


@router.post("/chat/reset")
async def reset_chat(session_id: str = "default"):
    SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}