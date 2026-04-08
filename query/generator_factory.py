# TODO: implement in:
# query/generator.py  (switchable version)

import os

def get_generator(backend: str = "ollama", **kwargs):
    if backend == "anthropic":
        from query.backends.anthropic_gen import AnthropicGenerator
        return AnthropicGenerator(**kwargs)
    elif backend == "ollama":
        from query.backends.ollama_gen import OllamaGenerator
        return OllamaGenerator(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")
