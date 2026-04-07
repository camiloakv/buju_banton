from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion.chunker import Document


class Embedder:
    """
    Wraps a sentence-transformers model to embed Document chunks.

    model_name options (tradeoff: quality vs speed):
      - "all-MiniLM-L6-v2"      → fast, lightweight, good baseline  (~80MB)
      - "all-mpnet-base-v2"     → stronger quality, slower           (~420MB)
      - "BAAI/bge-large-en-v1.5"→ near-SOTA open source             (~1.3GB)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.dimension}")

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        """
        Embed a list of Documents. Returns a (N, dim) float32 array.
        Preserves order — index i in output matches index i in input.
        """
        texts = [doc.content for doc in docs]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,    # cosine similarity = dot product
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns a (dim,) float32 vector."""
        vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return vector.astype(np.float32)
