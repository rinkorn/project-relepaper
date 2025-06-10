from relepaper.config.settings import AppSettings, ChatModelSettings, OllamaSettings
from pathlib import Path


def get_dev_settings() -> AppSettings:
    """Return development settings."""
    return AppSettings(
        project_path=Path("/home/rinkorn/space/prog/python/sber/project-relepaper"),
        log_dir="logs",
        chat_model=ChatModelSettings(platform="ollama", model_name="qwen3:4b", temperature=0.0, max_tokens=10000),
        ollama=OllamaSettings(model="qwen3:4b", temperature=0.0, max_tokens=10000, host="http://localhost:11434"),
    )
