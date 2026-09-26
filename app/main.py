# app/main.py
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import documents, chat
from app.routes import graph_chat      # Step 8, experimental

app = FastAPI(
    title="RAG Chatbot API",
    description="Document-grounded chatbot built with FastAPI and OpenAI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for local development
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(graph_chat.router)   # Step 8, experimental: 403 unless GRAPH_RAG_ENABLED


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the frontend. Must stay LAST: a mount at "/" matches every path,
# so anything registered after it would never be reached.
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")