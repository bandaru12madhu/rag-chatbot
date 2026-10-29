import pytest

from app.llm import get_llm


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm(provider="not-a-provider")
