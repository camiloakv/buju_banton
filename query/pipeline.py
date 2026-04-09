from dataclasses import dataclass
from typing import List, Tuple

from ingestion.chunker import Document
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from query.retriever import Retriever
from query.prompt_builder import PromptBuilder
from query.generator import Generator


@dataclass
class RAGResponse:
    query:     str
    answer:    str
    sources:   List[Tuple[Document, float]]   # [(doc, score), ...]


class RAGPipeline:
    """
    End-to-end RAG: query → retrieve → prompt → generate → response.

    Available models:
        [ ] "fast": llama3.2, 3B  — low RAM, quick iteration
        [X] "balanced": llama3.1, 8B  — recommended default
        [ ] "reasoning": mistral, 7B  — strong on complex questions
        [ ] "long-context": mistral-nemo, 12B — large retrieved contexts
        [ ] "structured": gemma2, 9B  — citation-style outputs
        [ ] "tiny": phi3, 3.8B — minimal footprint
        [X] "multilingual": qwen2.5, 7B  — non-English docs
    """

    def __init__(
        self,
        embedder:    Embedder,
        store:       VectorStore,
        top_k:       int = 5,
        max_context: int = 3000,
        model:       str = "balanced",   # ← was "claude-haiku-..."
    ):
        self.retriever      = Retriever(embedder, store, top_k=top_k)
        self.prompt_builder = PromptBuilder(max_context_chars=max_context)
        self.generator      = Generator(model=model)

    def run(self, query: str) -> RAGResponse:
        """Batch mode — returns a complete RAGResponse."""

        # 1. Retrieve
        results = self.retriever.retrieve(query)

        # 2. Build prompt
        system_prompt, user_message = self.prompt_builder.build(query, results)

        # 3. Generate
        answer = self.generator.generate(system_prompt, user_message)

        return RAGResponse(query=query, answer=answer, sources=results)

    def stream(self, query: str):
        """
        Streaming mode — prints answer token by token, returns RAGResponse.
        Useful for interactive demos and FastAPI streaming endpoints.
        """
        results = self.retriever.retrieve(query)
        system_prompt, user_message = self.prompt_builder.build(query, results)

        print(f"\nQ: {query}\nA: ", end="", flush=True)
        chunks = []
        for chunk in self.generator.stream(system_prompt, user_message):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        print()

        return RAGResponse(
            query   = query,
            answer  = "".join(chunks),
            sources = results,
        )
