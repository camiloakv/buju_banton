import requests
import json
from typing import Iterator


OLLAMA_MODELS = {
    "fast":        "llama3.2",        # 3B  — low RAM, quick iteration
    "balanced":    "llama3.1",        # 8B  — recommended default
    "reasoning":   "mistral",         # 7B  — strong on complex questions
    "long-context":"mistral-nemo",    # 12B — large retrieved contexts
    "structured":  "gemma2",          # 9B  — citation-style outputs
    "tiny":        "phi3",            # 3.8B — minimal footprint
    "multilingual":"qwen2.5",         # 7B  — non-English docs
}


class Generator:
    """
    Calls a local Ollama model for answer generation.

    Usage:
        gen = Generator(model="balanced")          # alias
        gen = Generator(model="llama3.1")          # or direct name
    """

    def __init__(self, model: str = "balanced", max_tokens: int = 512):
        # resolve alias → actual model name
        self.model      = OLLAMA_MODELS.get(model, model)
        self.max_tokens = max_tokens
        self.base_url   = "http://localhost:11434/api"
        self._check_connection()

    def _check_connection(self) -> None:
        """Fail fast if Ollama isn't running or the model isn't pulled."""
        try:
            resp = requests.get(f"{self.base_url}/tags", timeout=3)
            resp.raise_for_status()
            available = [m["name"].split(":")[0] for m in resp.json()["models"]]
            if self.model not in available:
                raise RuntimeError(
                    f"Model '{self.model}' not found locally.\n"
                    f"Run:  ollama pull {self.model}\n"
                    f"Available: {available}"
                )
        except requests.ConnectionError:
            raise RuntimeError(
                "Ollama is not running. Start it with:  ollama serve"
            )

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Single-shot generation — returns full answer string."""
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "model":   self.model,
                "stream":  False,
                "options": {"num_predict": self.max_tokens},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        """Streaming generation — yields text chunks as they arrive."""
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "model":   self.model,
                "stream":  True,
                "options": {"num_predict": self.max_tokens},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
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
