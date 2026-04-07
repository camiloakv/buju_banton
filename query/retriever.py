from typing import List, Tuple
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from ingestion.chunker import Document


class Retriever:
    """
    Embeds a query and fetches the top-k most relevant chunks.
    Thin layer now — designed to be extended with re-ranking or hybrid search.
    """

    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int=5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """Returns [(Document, similarity_score), ...] sorted by relevance."""
        query_vector = self.embedder.embed_query(query)
        return self.store.search(query_vector, top_k=self.top_k)
