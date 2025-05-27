import abc
from typing import Any, Callable, Dict

from relepaper.config.settings import AppSettings, load_settings
from relepaper.domains.langgraph.entities.chat_model_config import ChatModelConfig
from relepaper.domains.langgraph.external.chat_model.factory import ChatModelAdapterFactory
from relepaper.domains.langgraph.services.chat_model_service import ChatModelService
from relepaper.presenters.console_presenter import ConsolePresenter
from relepaper.presenters.gradio_presenter import GradioPresenter
from relepaper.views.cli.console_view import ConsoleView
from relepaper.views.gradio.gradio_view import GradioView


class IContainer(abc.ABC):
    """
    Интерфейс для контейнера IoC.
    """

    @abc.abstractmethod
    def register(self, name: str, factory: Callable):
        pass

    @abc.abstractmethod
    def resolve(self, name: str):
        pass

    @abc.abstractmethod
    def register_singleton(self, name: str, factory: Callable):
        pass


class CoreContainer(IContainer):
    """
    Контейнер IoC для управления зависимостями.
    """

    def __init__(self):
        self._dependencies: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._singleton_factories: Dict[str, Callable] = {}

    def register(self, name: str, factory: Callable):
        """Регистрирует зависимость с фабрикой."""
        self._dependencies[name] = factory

    def register_singleton(self, name: str, factory: Callable):
        """Регистрирует синглтон зависимость."""
        self._singleton_factories[name] = factory

    def resolve(self, name: str):
        """Разрешает зависимость."""
        # Сначала проверяем синглтоны
        if name in self._singleton_factories:
            if name not in self._singletons:
                self._singletons[name] = self._singleton_factories[name]()
            return self._singletons[name]

        # Затем обычные зависимости
        if name in self._dependencies:
            return self._dependencies[name]()

        raise ValueError(f"Dependency '{name}' not found")


def configure_container(settings: AppSettings = None) -> CoreContainer:
    """Конфигурирует контейнер с настройками."""
    if settings is None:
        settings = load_settings()

    container = CoreContainer()

    # Регистрируем настройки как синглтон
    container.register_singleton("settings", lambda: settings)

    # Создаем конфигурацию адаптеров на основе настроек
    def get_adapter_config(platform: str) -> Dict[str, Any]:
        match platform:
            case "ollama":
                return {
                    "model": settings.ollama.model,
                    "temperature": settings.ollama.temperature,
                    "max_tokens": settings.ollama.max_tokens,
                    "host": settings.ollama.host,
                }
            case "gigachat":
                return {
                    "credentials": settings.gigachat.credentials,
                    "scope": settings.gigachat.scope,
                    "verify_ssl_certs": settings.gigachat.verify_ssl_certs,
                }
            case "lmstudio":
                return {
                    "base_url": settings.lmstudio.base_url,
                    "api_key": settings.lmstudio.api_key,
                    "temperature": settings.lmstudio.temperature,
                    "max_tokens": settings.lmstudio.max_tokens,
                }
            case _:
                raise ValueError(f"Unknown platform: {platform}")

    # Регистрируем конфигурацию чат-модели
    container.register_singleton(
        "chat_model_config",
        lambda: ChatModelConfig(
            platform=settings.chat_model.platform,
            model_name=settings.chat_model.model_name,
            temperature=settings.chat_model.temperature,
            max_tokens=settings.chat_model.max_tokens,
            additional_params=settings.chat_model.additional_params,
        ),
    )

    # Регистрируем адаптер
    container.register_singleton(
        "chat_model_adapter",
        lambda: ChatModelAdapterFactory.create_adapter(
            settings.chat_model.platform,
            get_adapter_config(settings.chat_model.platform),
        ),
    )

    # Регистрируем готовую чат-модель
    container.register_singleton(
        "chat_model",
        lambda: container.resolve("chat_model_adapter").create(),
    )

    # Регистрируем сервис
    container.register_singleton(
        "chat_model_service",
        lambda: ChatModelService(container.resolve("chat_model")),
    )

    # Регистрируем представления
    container.register("console_view", lambda: ConsoleView())
    container.register("gradio_view", lambda: GradioView())

    # Регистрируем презентеры
    container.register(
        "console_presenter",
        lambda: ConsolePresenter(
            container.resolve("console_view"),
            container.resolve("chat_model_service"),
        ),
    )
    container.register(
        "gradio_presenter",
        lambda: GradioPresenter(
            container.resolve("gradio_view"),
            container.resolve("chat_model_service"),
        ),
    )

    return container
