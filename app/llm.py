"""LLM abstraction layer using Foundry Local SDK."""

from __future__ import annotations

from typing import Optional
from foundry_local_sdk import Configuration, FoundryLocalManager

DEFAULT_MODEL_NAME = "qwen3-4b"
DEFAULT_APP_NAME = "foundry"


class LocalLLM:
    """Interface for local LLM inference via Foundry Local SDK."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        app_name: str = DEFAULT_APP_NAME,
    ) -> None:
        self.model_name = model_name
        self.app_name = app_name
        self._ensure_initialized()
        self._chat_client = None

    def _ensure_initialized(self) -> None:
        """Initialize FoundryLocalManager singleton if not already initialized."""
        if FoundryLocalManager.instance is None:
            config = Configuration(app_name=self.app_name)
            FoundryLocalManager.initialize(config)

    def _get_chat_client(self):
        """Retrieve and cache the chat completions client for the configured model."""
        if self._chat_client is None:
            manager = FoundryLocalManager.instance
            if manager is None:
                raise RuntimeError("FoundryLocalManager is not initialized.")
            model = manager.catalog.get_model(self.model_name)
            if model is None:
                raise ValueError(
                    f"Model '{self.model_name}' was not found in the Foundry Local catalog."
                )
            if not model.is_loaded:
                model.load()
            self._chat_client = model.get_chat_client()
        return self._chat_client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = 512,
        temperature: Optional[float] = 0.7,
    ) -> str:
        """Send a prompt to the local model and return the generated text response."""
        client = self._get_chat_client()
        if max_tokens is not None:
            client.settings.max_tokens = max_tokens
        if temperature is not None:
            client.settings.temperature = temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.complete_chat(messages)
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content or ""
        return ""


def generate_response(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = 512,
    temperature: Optional[float] = 0.7,
) -> str:
    """Convenience function to generate a response from the local LLM."""
    llm = LocalLLM(model_name=model_name)
    return llm.generate(
        prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

