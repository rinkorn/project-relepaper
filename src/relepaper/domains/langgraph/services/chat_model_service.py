from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from relepaper.domains.langgraph.services.interfaces import IService


class ChatModelService(IService):
    """Доменный сервис для работы с чат-моделями."""

    def __init__(self, chat_model: BaseChatModel):
        self._chat_model = chat_model

    def invoke(self, messages: str | BaseMessage | List[BaseMessage]) -> BaseMessage:
        """Отправляет сообщения в чат-модель и возвращает ответ."""
        response = self._chat_model.invoke(messages)
        return response

    def stream(self, messages: List[BaseMessage]):
        """Стримит ответ от чат-модели."""
        return self._chat_model.stream(messages)

    def batch(self, messages_list: List[List[BaseMessage]]) -> List[str]:
        """Обрабатывает несколько запросов батчем."""
        responses = self._chat_model.batch(messages_list)
        return [response.content for response in responses]

    def get_model_name(self) -> str:
        """Возвращает название модели."""
        return getattr(self._chat_model, "model_name", "unknown")

    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели."""
        return {
            "model_name": self.get_model_name(),
            "model_type": type(self._chat_model).__name__,
            # Добавьте другие полезные свойства модели
        }
