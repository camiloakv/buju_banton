import requests
import json
from typing import Iterator


class Generator:
    """
    Calls a local Ollama model instead of the Anthropic API.

    Setup:
        1. Install Ollama: https://ollama.com
        2. Pull a model:   ollama pull llama3.2
        3. It runs on http://localhost:11434 by default
    """

    def __init__(self, model: str = "llama3.2", max_tokens: int = 512):
        self.model      = model
        self.max_tokens = max_tokens
        self.base_url   = "http://localhost:11434/api"

    def generate(self, system_prompt: str, user_message: str) -> str:
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "model":    self.model,
                "stream":   False,
                "options":  {"num_predict": self.max_tokens},
                "messages": [
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": user_message},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "model":    self.model,
                "stream":   True,
                "options":  {"num_predict": self.max_tokens},
                "messages": [
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": user_message},
                ],
            },
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if not chunk.get("done"):
                    yield chunk["message"]["content"]
