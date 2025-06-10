from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ChatModelConfig:
    """Chat model configuration."""

    platform: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1000
    additional_params: Optional[Dict[str, Any]] = None
