import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
from ingestion.chunker import Document


class VectorStore:
    """
    FAISS-backed vector store. Stores embeddings + Document metadata together
    so retrieval returns full Document objects, not just indices.

    Index type: IndexFlatIP (exact cosine similarity on normalized vectors).
    Persistence: saves/loads both the FAISS index and the doc list to disk.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)   # IP = inner product
        self.documents: List[Document] = []          # parallel list to index

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add(self, docs: List[Document], embeddings: np.ndarray) -> None:
        """
        Add documents and their embeddings to the store.
        embeddings must be shape (N, dimension), float32, L2-normalized.
        """
        assert embeddings.shape == (len(docs), self.dimension), (
            f"Shape mismatch: got {embeddings.shape}, "
            f"expected ({len(docs)}, {self.dimension})"
        )
        assert embeddings.dtype == np.float32, "Embeddings must be float32"

        self.index.add(embeddings)
        self.documents.extend(docs)
        print(f"  Indexed {len(docs)} chunks — total: {self.index.ntotal}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Return the top_k most similar Documents and their similarity scores.
        query_vector must be shape (dimension,), float32, L2-normalized.

        Returns: list of (Document, score) sorted descending by score.
        Score range: [-1, 1] — higher is more similar.
        """
        query_vector = query_vector.reshape(1, -1)   # FAISS expects (1, dim)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:                            # FAISS pads with -1 if <k results
                continue
            results.append((self.documents[idx], float(score)))

        return results                               # already sorted by FAISS

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, dir_path: str | Path) -> None:
        """Persist index and documents to disk."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(dir_path / "index.faiss"))

        with open(dir_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        print(f"Saved {self.index.ntotal} vectors → {dir_path}")

    @classmethod
    def load(cls, dir_path: str | Path) -> "VectorStore":
        """Load a previously saved VectorStore from disk."""
        dir_path = Path(dir_path)

        index = faiss.read_index(str(dir_path / "index.faiss"))

        with open(dir_path / "documents.pkl", "rb") as f:
            documents = pickle.load(f)

        store = cls(dimension=index.d)
        store.index = index
        store.documents = documents

        print(f"Loaded {index.ntotal} vectors from {dir_path}")
        return store
