from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from relepaper.domains.langgraph.external.interfaces import IChatModelAdapter


class LMStudioAdapter(IChatModelAdapter):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ):
        self._base_url = base_url
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs

    def create(self) -> BaseChatModel:
        return ChatOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **self._kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "base_url": self._base_url,
            "api_key": self._api_key,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            **self._kwargs,
        }
