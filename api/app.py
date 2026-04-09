from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import query, health
from api.dependencies import get_pipeline


def create_app() -> FastAPI:
    app = FastAPI(
        title       = "RAG Pipeline API",
        description = "Retrieval-Augmented Generation over your document corpus",
        version     = "0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    app.include_router(query.router)
    app.include_router(health.router)

    @app.on_event("startup")
    async def startup():
        """Pre-load the pipeline at startup so first request isn't slow."""
        get_pipeline()
        print("Pipeline loaded and ready.")

    return app


app = create_app()
