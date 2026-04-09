from functools import lru_cache
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from query.pipeline import RAGPipeline
from config import settings             # we'll define this next


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore.load(settings.vector_store_path)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(
        embedder    = get_embedder(),
        store       = get_vector_store(),
        top_k       = settings.default_top_k,
        model       = settings.llm_model,
        max_context = settings.max_context_chars,
    )
