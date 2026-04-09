from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import json

from api.schemas import QueryRequest, QueryResponse, SourceDocument
from api.dependencies import get_pipeline
from query.pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["query"])


def _format_sources(response) -> list[SourceDocument]:
    return [
        SourceDocument(
            source      = doc.source,
            chunk_index = doc.chunk_index,
            score       = round(score, 4),
            preview     = doc.content[:200],
        )
        for doc, score in response.sources
    ]


@router.post("", response_model=QueryResponse)
async def query_batch(
    request:  QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Single-shot query — waits for the full answer before returning."""
    try:
        response = pipeline.run(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question = request.question,
        answer   = response.answer,
        sources  = _format_sources(response),
    )


@router.get("/stream")
async def query_stream(
    question: str,
    top_k:    int      = 5,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Streaming query via Server-Sent Events.
    Each token arrives as a 'token' event.
    A final 'done' event carries the source documents as JSON.

    Usage:
        curl -N "http://localhost:8000/query/stream?question=What+is+X"
    """
    async def event_generator():
        try:
            # Retrieve once, then stream generation
            results = pipeline.retriever.retrieve(question)
            system_prompt, user_message = pipeline.prompt_builder.build(
                question, results
            )

            # Stream tokens
            for token in pipeline.generator.stream(system_prompt, user_message):
                yield {"event": "token", "data": token}

            # Final event — sources as JSON payload
            sources = [
                {
                    "source":      doc.source,
                    "chunk_index": doc.chunk_index,
                    "score":       round(score, 4),
                    "preview":     doc.content[:200],
                }
                for doc, score in results
            ]
            yield {"event": "done", "data": json.dumps(sources)}

        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())
