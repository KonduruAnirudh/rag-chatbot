# app/rag/chunk.py
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

# Separators tried in order of preference when looking for a clean break.
SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_text(
    text: str,
    doc_id: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping chunks that end on natural boundaries.
    Returns a list of dicts, each with its text and its provenance.
    """
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # If this isn't the final chunk, back up to a natural boundary.
        if end < len(text):
            end = _find_break(text, start, end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": index,
            })
            index += 1

        # Step forward, minus the overlap.
        next_start = end - overlap
        # Guard against never advancing on pathological input.
        start = next_start if next_start > start else end

    return chunks


def _find_break(text: str, start: int, end: int) -> int:
    """
    Look backwards from `end` for a separator, preferring paragraph
    breaks over line breaks over sentence ends over spaces.
    Only searches the last 20% of the chunk so chunks stay near target size.
    """
    window = max(start, end - int((end - start) * 0.2))

    for sep in SEPARATORS:
        pos = text.rfind(sep, window, end)
        if pos != -1:
            return pos + len(sep)

    return end  # no boundary found; cut at the target