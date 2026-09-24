# app/rag/chunk.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

# Tried in order: paragraph, sentence, line, word, then character.
# Sentence comes BEFORE line: in PDF-extracted text a line break is just
# where the page wrapped visually, usually mid-sentence.
SEPARATORS = ["\n\n", ". ", "\n", " ", ""]

PROBE_LENGTH = 60   # enough text to locate a chunk uniquely in the document


def chunk_text(
    text: str,
    doc_id: str,
    filename: str,
    page_spans: list[dict] | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping chunks, recording which pages each one covers.

    Chunks may span a page boundary — that is deliberate, so a sentence split
    across pages stays intact — which is why each chunk gets a page RANGE
    rather than a single page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        keep_separator="end",   # ". " stays on the sentence it ends
    )

    chunks = []
    cursor = 0
    for piece in splitter.split_text(text):
        piece = piece.strip()
        if not piece or _is_noise(piece):
            continue

        start = _locate(text, piece, cursor)
        cursor = start + 1          # chunks overlap, so advance by one character
        page_start, page_end, from_ocr = _pages_for(page_spans, start, start + len(piece))

        chunks.append({
            "text": piece,
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": len(chunks),
            "page_start": page_start,
            "page_end": page_end,
            "ocr": from_ocr,
        })
    return chunks


def _locate(text: str, chunk: str, cursor: int) -> int:
    """
    Find where a chunk sits in the original text.

    The splitter returns strings, not positions, so we search for the chunk's
    opening characters. Searching forward from the last match keeps this linear
    and avoids matching a repeated header earlier in the document.
    """
    probe = chunk[:PROBE_LENGTH]
    index = text.find(probe, cursor)
    if index == -1:
        index = text.find(probe)        # stripping may have shifted it; try globally
    return index if index != -1 else cursor


def _pages_for(spans, start: int, end: int) -> tuple[int | None, int | None, bool]:
    """Which pages does the character range [start, end) overlap?"""
    if not spans:
        return None, None, False
    hits = [s for s in spans if s["start"] < end and s["end"] > start]
    if not hits:
        return None, None, False
    return hits[0]["page"], hits[-1]["page"], any(h["ocr"] for h in hits)


def _is_noise(text: str) -> bool:
    """Table-of-contents pages extract as dot leaders and page numbers."""
    stripped = text.replace(".", "").replace(" ", "").replace("\n", "")
    return len(stripped) < len(text) * 0.4