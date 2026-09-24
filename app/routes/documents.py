# app/routes/documents.py

import hashlib
import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.rag.extract import extract_text, ExtractionError
from app.rag.chunk import chunk_text
from app.rag.embed import embed_texts, EmbeddingError
from app.rag.store import store
from app.rag.registry import registry


router = APIRouter(prefix="/api", tags=["documents"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    # 1. Validate the file extension
    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: PDF, TXT.",
        )

    # 2. Read the file and validate its size
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large. Maximum size is "
                f"{MAX_FILE_SIZE // (1024 * 1024)} MB."
            ),
        )

    # 2b. Reject exact duplicate files
    content_hash = hashlib.sha256(contents).hexdigest()

    existing = registry.find_by_hash(content_hash)

    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This file has already been uploaded as "
                f"'{existing['filename']}' "
                f"(doc_id: {existing['doc_id']})."
            ),
        )

    # 3. Generate a unique document ID and storage path
    #    Never trust the client's filename for storage.
    doc_id = str(uuid.uuid4())
    stored_path = UPLOAD_DIR / f"{doc_id}{suffix}"

    try:
        stored_path.write_bytes(contents)
    except OSError as e:
        print(f"[UPLOAD ERROR] {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not save the file.",
        )

    # 4. Extract text
    #    If extraction fails, remove the saved file.
    try:
        text, extract_report = await asyncio.to_thread(extract_text, stored_path)
    except ExtractionError as e:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=str(e),
        )

    # 5. Split the document into overlapping chunks
    chunks = chunk_text(
        text,
        doc_id=doc_id,
        filename=file.filename,
    )

    # 6. Generate embeddings for every chunk
    try:
        vectors = await embed_texts(
            [chunk["text"] for chunk in chunks]
        )
    except EmbeddingError as e:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=502,
            detail=str(e),
        )

    # 7. Add the vectors and chunk metadata to the persistent vector store
    store.add(vectors, chunks)

    # 8. Register the document metadata
    registry.add(
        doc_id=doc_id,
        filename=file.filename,
        path=str(stored_path),
        size_bytes=len(contents),
        char_count=len(text),
        chunk_count=len(chunks),
        content_hash=content_hash,
    )

    # 9. Return upload/indexing information
    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "size_bytes": len(contents),
        "char_count": len(text),
        "chunk_count": len(chunks),
        "ocr_pages": extract_report["ocr_pages"],
        "total_pages": extract_report["total_pages"],
        "ocr_skipped": extract_report["ocr_skipped"],
        "total_chunks_indexed": store.count(),
        "message": "Uploaded and processed.",
    }


@router.get("/documents")
async def list_documents():
    return {
        "count": registry.count(),
        "total_chunks_indexed": store.count(),
        "documents": [
            {
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "size_bytes": d["size_bytes"],
                "chunk_count": d["chunk_count"],
                "uploaded_at": d["uploaded_at"],
            }
            for d in registry.list_all()
        ],
    }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    # 1. Find the document in the registry
    record = registry.get(doc_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    # 2. Remove its vectors from the vector store
    removed = store.delete_document(doc_id)

    # 3. Remove the physical file
    Path(record["path"]).unlink(missing_ok=True)

    # 4. Remove the document from the registry
    registry.remove(doc_id)

    return {
        "doc_id": doc_id,
        "chunks_removed": removed,
    }