from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_gigachat import GigaChat

from relepaper.domains.langgraph.external.interfaces import IChatModelAdapter


class GigachatAdapter(IChatModelAdapter):
    def __init__(
        self,
        credentials: str,
        scope: str,
        verify_ssl_certs: bool = False,
        **kwargs: Any,
    ):
        self._credentials = credentials
        self._scope = scope
        self._verify_ssl_certs = verify_ssl_certs
        self._kwargs = kwargs

    def create(self) -> BaseChatModel:
        return GigaChat(
            credentials=self._credentials,
            scope=self._scope,
            verify_ssl_certs=self._verify_ssl_certs,
            **self._kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "platform": "gigachat",
            "credentials": self._credentials,
            "scope": self._scope,
            "verify_ssl_certs": self._verify_ssl_certs,
            **self._kwargs,
        }
