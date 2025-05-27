from pathlib import Path

from relepaper.config.settings import AppSettings, ChatModelSettings
from relepaper.containers.core import configure_container


def test_container_with_custom_settings():
    """Тест контейнера с кастомными настройками."""
    test_settings = AppSettings(
        project_path=Path("/tmp/test"),
        chat_model=ChatModelSettings(platform="ollama", model_name="test_model", temperature=0.5, max_tokens=1000),
    )

    container = configure_container(test_settings)

    chat_model_config = container.resolve("chat_model_config")
    assert chat_model_config.platform == "ollama"
    assert chat_model_config.model_name == "test_model"
    assert chat_model_config.temperature == 0.5
