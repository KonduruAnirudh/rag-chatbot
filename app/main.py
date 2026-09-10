# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents          
from app.routes import documents, chat

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
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}