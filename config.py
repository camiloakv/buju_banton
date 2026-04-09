from pydantic_settings import BaseSettings     # pip install pydantic-settings


class Settings(BaseSettings):
    embedding_model:   str = "all-MiniLM-L6-v2"
    vector_store_path: str = "./vector_store"
    llm_model:         str = "balanced"         # resolves to llama3.1
    default_top_k:     int = 5
    max_context_chars: int = 3000
    ollama_base_url:   str = "http://localhost:11434"

    model_config = {"env_file": ".env"}         # overridable via .env


settings = Settings()
