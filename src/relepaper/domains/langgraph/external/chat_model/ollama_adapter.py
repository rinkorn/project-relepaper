import os
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from relepaper.domains.langgraph.external.interfaces import IChatModelAdapter


class OllamaAdapter(IChatModelAdapter):
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        host: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._host = host
        self._kwargs = kwargs
        # Устанавливаем переменную окружения при инициализации
        os.environ["OLLAMA_HOST"] = self._host

    def create(self) -> BaseChatModel:
        return ChatOllama(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **self._kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "host": self._host,
            **self._kwargs,
        }
