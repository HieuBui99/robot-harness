"""Explicit provider selection; model capabilities are never guessed by name."""

import os

from .base import Provider


def create_provider(provider: str, model: str, api_key: str | None = None,
                    base_url: str | None = None) -> Provider:
    if not model or not model.strip():
        raise ValueError("Select a model with --model or LLM_MODEL")
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY or --api-key (use an explicit dummy key for an unauthenticated local server)")
        return OpenAIProvider(model, key, base_url or os.environ.get("OPENAI_BASE_URL"))
    if provider == "gemini":
        from .gemini_provider import GeminiProvider

        if base_url:
            raise ValueError("--base-url is for the OpenAI-compatible provider, not native Gemini")
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY or --api-key")
        return GeminiProvider(model, key)
    raise ValueError(f"Unknown provider: {provider}")
