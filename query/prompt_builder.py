from typing import List, Tuple
from ingestion.chunker import Document


SYSTEM_PROMPT = """You are a helpful assistant that answers questions \
based strictly on the provided context passages.

Rules:
- Only use information present in the context below.
- If the context does not contain enough information, say so clearly.
- When possible, indicate which source your answer draws from.
- Be concise and factual."""


class PromptBuilder:
    """
    Assembles retrieved chunks into a structured prompt for the LLM.

    Context format:
        [1] source: file.pdf — chunk 3
        <chunk text>

        [2] source: other.txt — chunk 7
        <chunk text>
        ...
    Then appends the user question.
    """

    def __init__(self, max_context_chars: int = 3000):
        self.max_context_chars = max_context_chars  # stay within LLM context window

    def build(
        self,
        query: str,
        results: List[Tuple[Document, float]],
    ) -> Tuple[str, str]:
        """
        Returns (system_prompt, user_message) ready to pass to the LLM.
        Truncates context if it exceeds max_context_chars.
        """
        context_blocks = []
        total_chars    = 0

        for i, (doc, score) in enumerate(results, start=1):
            block = (
                f"[{i}] source: {doc.source} — chunk {doc.chunk_index}\n"
                f"{doc.content}"
            )
            if total_chars + len(block) > self.max_context_chars:
                break  # hard cap to avoid overflows
            context_blocks.append(block)
            total_chars += len(block)

        context_str = "\n\n".join(context_blocks)

        user_message = (
            f"Context passages:\n\n{context_str}\n\n"
            f"---\n\nQuestion: {query}"
        )

        return SYSTEM_PROMPT, user_message
