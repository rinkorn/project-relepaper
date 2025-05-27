from typing import Any, Dict

from relepaper.domains.langgraph.external.chat_model.gigachat_adapter import GigachatAdapter
from relepaper.domains.langgraph.external.chat_model.lmstudio_adapter import LMStudioAdapter
from relepaper.domains.langgraph.external.chat_model.ollama_adapter import OllamaAdapter
from relepaper.domains.langgraph.external.interfaces import IChatModelAdapter


class ChatModelAdapterFactory:
    """Фабрика для создания адаптеров чат-моделей."""

    @staticmethod
    def create_adapter(platform: str, config: Dict[str, Any]) -> IChatModelAdapter:
        """Создает словарь адаптеров на основе конфигурации."""

        match platform:
            case "gigachat":
                return GigachatAdapter(**config)
            case "ollama":
                return OllamaAdapter(**config)
            case "lmstudio":
                return LMStudioAdapter(**config)
            case _:
                raise ValueError(f"Unknown platform: {platform}")
