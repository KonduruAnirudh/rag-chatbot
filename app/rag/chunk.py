# app/rag/chunk.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

# Tried in order: paragraph, sentence, line, word, then character.
# Sentence comes BEFORE line: in PDF-extracted text a line break is just
# where the page wrapped visually, usually mid-sentence.
SEPARATORS = ["\n\n", ". ", "\n", " ", ""]


def chunk_text(
    text: str,
    doc_id: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping chunks using LangChain's recursive splitter.
    Returns a list of dicts, each with its text and its provenance.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        keep_separator="end",   # ". " stays on the sentence it ends, not the next one
    )

    chunks = []
    for piece in splitter.split_text(text):
        piece = piece.strip()
        if not piece or _is_noise(piece):
            continue
        chunks.append({
            "text": piece,
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": len(chunks),
        })
    return chunks


def _is_noise(text: str) -> bool:
    """Table-of-contents pages extract as dot leaders and page numbers."""
    stripped = text.replace(".", "").replace(" ", "").replace("\n", "")
    return len(stripped) < len(text) * 0.4