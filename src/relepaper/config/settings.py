import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ChatModelSettings:
    """Chat model settings."""

    platform: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 10000
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class OllamaSettings:
    """Ollama settings."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 10000
    host: str = "http://localhost:11434"


@dataclass
class GigachatSettings:
    """Gigachat settings."""

    credentials: str
    scope: str = "GIGACHAT_API_PERS"
    verify_ssl_certs: bool = False


@dataclass
class LMStudioSettings:
    """LMStudio settings."""

    base_url: str = "http://localhost:7007/v1"
    api_key: str = "not_needed"
    temperature: float = 0.0
    max_tokens: int = 10000


@dataclass
class AppSettings:
    """Application settings."""

    project_path: Path
    log_dir: Path = Path("logs")

    # Chat model settings
    chat_model: ChatModelSettings = field(
        default_factory=lambda: ChatModelSettings(
            platform=os.getenv("CHAT_MODEL_PLATFORM", "ollama"),
            model_name=os.getenv("CHAT_MODEL_NAME", "qwen3:4b"),
            temperature=float(os.getenv("CHAT_MODEL_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("CHAT_MODEL_MAX_TOKENS", "10000")),
        )
    )

    # Platform settings
    ollama: OllamaSettings = field(
        default_factory=lambda: OllamaSettings(
            model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "10000")),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    )

    gigachat: GigachatSettings = field(
        default_factory=lambda: GigachatSettings(
            credentials=os.getenv("GIGACHAT_CREDENTIALS", ""),
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true",
        )
    )

    lmstudio: LMStudioSettings = field(
        default_factory=lambda: LMStudioSettings(
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:7007/v1"),
            api_key=os.getenv("LMSTUDIO_API_KEY", "not_needed"),
            temperature=float(os.getenv("LMSTUDIO_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LMSTUDIO_MAX_TOKENS", "10000")),
        )
    )


def load_settings() -> AppSettings:
    """Load application settings."""
    _default_project_path = Path(__file__).resolve().parents[3]
    project_path = Path(os.getenv("PROJECT_PATH", _default_project_path))

    return AppSettings(project_path=project_path)
