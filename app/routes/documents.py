# app/routes/documents.py
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.rag.extract import extract_text, ExtractionError
from app.rag.chunk import chunk_text
from app.rag.embed import embed_texts, EmbeddingError
from app.rag.store import store

router = APIRouter(prefix="/api", tags=["documents"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# In-memory registry of uploaded documents.
# Note: this resets on restart while the vector store persists —
# a known limitation, see README.
DOCUMENTS: dict[str, dict] = {}


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    # 1. Validate the extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: PDF, TXT.",
        )

    # 2. Read the bytes and check size
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB.",
        )

    # 3. Store under a generated name — never trust the client's filename
    doc_id = str(uuid.uuid4())
    stored_path = UPLOAD_DIR / f"{doc_id}{suffix}"

    try:
        stored_path.write_bytes(contents)
    except OSError as e:
        print(f"[UPLOAD ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Could not save the file.")

    # 4. Extract text — fail now rather than indexing an unusable document
    try:
        text = extract_text(stored_path)
    except ExtractionError as e:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    # 5. Split into overlapping chunks
    chunks = chunk_text(text, doc_id=doc_id, filename=file.filename)

    # 6. Embed every chunk
    try:
        vectors = await embed_texts([c["text"] for c in chunks])
    except EmbeddingError as e:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=str(e))

    # 7. Index them — this is what makes the document searchable
    store.add(vectors, chunks)

    # 8. Register the document (metadata only; the store owns the content)
    DOCUMENTS[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "path": str(stored_path),
        "size_bytes": len(contents),
        "char_count": len(text),
        "chunk_count": len(chunks),
    }

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "size_bytes": len(contents),
        "char_count": len(text),
        "chunk_count": len(chunks),
        "total_chunks_indexed": store.count(),
        "message": "Uploaded and processed.",
    }


@router.get("/documents")
async def list_documents():
    return {
        "count": len(DOCUMENTS),
        "total_chunks_indexed": store.count(),
        "documents": [
            {
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "size_bytes": d["size_bytes"],
                "chunk_count": d["chunk_count"],
            }
            for d in DOCUMENTS.values()
        ],
    }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found.")

    removed = store.delete_document(doc_id)
    Path(DOCUMENTS[doc_id]["path"]).unlink(missing_ok=True)
    del DOCUMENTS[doc_id]

    return {"doc_id": doc_id, "chunks_removed": removed}