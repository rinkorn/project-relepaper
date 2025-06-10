import abc
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel


class IAdapter(abc.ABC):
    """Base interface for all adapters."""

    pass


class IChatModelAdapter(abc.ABC):
    """Interface for chat model adapters."""

    @abc.abstractmethod
    def create(self) -> BaseChatModel:
        """Create an instance of a chat model."""
        pass

    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the adapter."""
        pass
