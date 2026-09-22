# app/rag/store.py
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.sparse import tokenize

INDEX_DIR = Path("data/index")
VECTORS_PATH = INDEX_DIR / "vectors.npy"
METADATA_PATH = INDEX_DIR / "metadata.json"


class VectorStore:
    """
    Holds every chunk three ways, aligned by position:
      self.metadata[i]  the chunk's text and provenance
      self.vectors[i]   its embedding             (dense search)
      BM25 document i   its keyword statistics    (sparse search)
    Only this class changes them, so they can never fall out of step.
    """

    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.metadata: list[dict] = []
        self.bm25: BM25Okapi | None = None
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
        self._rebuild_sparse()
        self._save()

    def dense_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of every chunk to the query (vectors are unit length)."""
        return self.vectors @ query_vector

    def sparse_scores(self, query: str) -> np.ndarray:
        """BM25 keyword score of every chunk for the query."""
        if self.bm25 is None:
            return np.zeros(len(self.metadata))
        return np.asarray(self.bm25.get_scores(tokenize(query)))

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        """Dense-only search, kept for diagnostics."""
        if self.is_empty():
            return []
        scores = self.dense_scores(query_vector)
        top = np.argsort(scores)[::-1][:min(top_k, len(scores))]
        return [{**self.metadata[i], "score": float(scores[i])} for i in top]

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
        self._rebuild_sparse()
        self._save()
        return removed

    def is_empty(self) -> bool:
        return self.vectors is None or len(self.metadata) == 0

    def count(self) -> int:
        return len(self.metadata)

    # --- internals --------------------------------------------------

    def _rebuild_sparse(self) -> None:
        """
        BM25 statistics are derived entirely from chunk text, so they are
        rebuilt rather than saved. That takes milliseconds at this scale, and
        a derived index that is never stored can never disagree with its source.
        """
        if self.metadata:
            self.bm25 = BM25Okapi([tokenize(m["text"]) for m in self.metadata])
        else:
            self.bm25 = None

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
            self._rebuild_sparse()


# Single shared instance, created once at import.
store = VectorStore()