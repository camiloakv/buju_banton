import anthropic
from typing import Iterator


class Generator:
    """
    Calls Claude via the Anthropic SDK to produce a grounded answer.
    Supports both batch (full response) and streaming modes.

    Set your API key:  export ANTHROPIC_API_KEY=sk-ant-...
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 512):
        self.client = anthropic.Anthropic()     # reads ANTHROPIC_API_KEY from env
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Single-shot generation. Returns the full answer string."""
        message = self.client.messages.create(
            model = self.model,
            max_tokens = self.max_tokens,
            system = system_prompt,
            messages = [{"role": "user", "content": user_message}],
        )
        return message.content[0].text

    def stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        """Streaming generation. Yields text chunks as they arrive."""
        with self.client.messages.stream(
            model = self.model,
            max_tokens = self.max_tokens,
            system = system_prompt,
            messages = [{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text
