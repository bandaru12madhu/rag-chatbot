"""Returns a chat model for the provider chosen in .env (openai or anthropic)."""
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings


def get_llm(provider: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    settings = get_settings()
    provider = (provider or settings.llm_provider).lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_model:
            raise ValueError("Set ANTHROPIC_MODEL in your .env file")

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic' or 'openai'.")
