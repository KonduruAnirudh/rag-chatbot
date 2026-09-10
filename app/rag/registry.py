# app/rag/registry.py
import json
from datetime import datetime, timezone
from pathlib import Path

INDEX_DIR = Path("data/index")
REGISTRY_PATH = INDEX_DIR / "documents.json"


class DocumentRegistry:
    """
    Persistent record of which documents exist in the knowledge base.
    Mirrors VectorStore's save/load pattern so the two share a lifetime.
    """

    def __init__(self):
        self.documents: dict[str, dict] = {}
        self._load()

    def add(self, doc_id: str, filename: str, path: str,
            size_bytes: int, char_count: int, chunk_count: int) -> dict:
        record = {
            "doc_id": doc_id,
            "filename": filename,
            "path": path,
            "size_bytes": size_bytes,
            "char_count": char_count,
            "chunk_count": chunk_count,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        self.documents[doc_id] = record
        self._save()
        return record

    def get(self, doc_id: str) -> dict | None:
        return self.documents.get(doc_id)

    def remove(self, doc_id: str) -> dict | None:
        record = self.documents.pop(doc_id, None)
        if record:
            self._save()
        return record

    def list_all(self) -> list[dict]:
        return list(self.documents.values())

    def count(self) -> int:
        return len(self.documents)

    # --- persistence ---

    def _save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(json.dumps(self.documents, indent=2))

    def _load(self) -> None:
        if REGISTRY_PATH.exists():
            self.documents = json.loads(REGISTRY_PATH.read_text())


registry = DocumentRegistry()