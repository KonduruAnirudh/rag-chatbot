# app/rag/store.py
import json
from pathlib import Path

import numpy as np

INDEX_DIR = Path("data/index")
VECTORS_PATH = INDEX_DIR / "vectors.npy"
METADATA_PATH = INDEX_DIR / "metadata.json"


class VectorStore:
    """
    Holds all chunk vectors in one matrix, with metadata aligned by row.
    Row i of `self.vectors` describes chunk `self.metadata[i]`.
    """

    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.metadata: list[dict] = []
        self._load()

    # --- public API -------------------------------------------------

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must be the same length")

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        self.metadata.extend(metadata)
        self._save()

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        """
        Return the top_k most similar chunks, each with a similarity score.
        Vectors are unit-length, so the dot product IS cosine similarity.
        """
        if self.is_empty():
            return []

        scores = self.vectors @ query_vector          # (n,) similarity scores
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]    # highest first

        return [
            {**self.metadata[i], "score": float(scores[i])}
            for i in top_indices
        ]

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to one document. Returns count removed."""
        if self.is_empty():
            return 0

        keep = [i for i, m in enumerate(self.metadata) if m["doc_id"] != doc_id]
        removed = len(self.metadata) - len(keep)
        if removed == 0:
            return 0

        self.vectors = self.vectors[keep] if keep else None
        self.metadata = [self.metadata[i] for i in keep]
        self._save()
        return removed

    def is_empty(self) -> bool:
        return self.vectors is None or len(self.metadata) == 0

    def count(self) -> int:
        return len(self.metadata)

    # --- persistence ------------------------------------------------

    def _save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if self.vectors is None:
            VECTORS_PATH.unlink(missing_ok=True)
            METADATA_PATH.unlink(missing_ok=True)
            return
        np.save(VECTORS_PATH, self.vectors)
        METADATA_PATH.write_text(json.dumps(self.metadata))

    def _load(self) -> None:
        if VECTORS_PATH.exists() and METADATA_PATH.exists():
            self.vectors = np.load(VECTORS_PATH)
            self.metadata = json.loads(METADATA_PATH.read_text())


# Single shared instance, created once at import.
store = VectorStore()