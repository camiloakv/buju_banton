import requests
from fastapi import APIRouter, Depends
from api.schemas import HealthResponse
from api.dependencies import get_vector_store, get_pipeline
from config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(store=Depends(get_vector_store)):
    """Returns liveness of Ollama and the vector store."""

    # Check Ollama
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    vector_store_ok = store.index.ntotal > 0

    return HealthResponse(
        status       = "ok" if (ollama_ok and vector_store_ok) else "degraded",
        ollama       = ollama_ok,
        vector_store = vector_store_ok,
        total_chunks = store.index.ntotal,
    )
