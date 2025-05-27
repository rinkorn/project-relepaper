import abc
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel


class IAdapter(abc.ABC):
    """Базовый интерфейс для всех адаптеров."""

    pass


class IChatModelAdapter(abc.ABC):
    """Интерфейс для адаптеров чат-моделей."""

    @abc.abstractmethod
    def create(self) -> BaseChatModel:
        """Создает экземпляр чат-модели."""
        pass

    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию адаптера."""
        pass
