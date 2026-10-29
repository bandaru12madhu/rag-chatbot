"""Central settings, loaded from environment variables / .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "openai"  # "openai" or "anthropic"

    anthropic_api_key: str = ""
    anthropic_model: str = ""  # required if llm_provider is anthropic

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_dir: str = "./data/chroma"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    upload_dir: str = "./data/uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
