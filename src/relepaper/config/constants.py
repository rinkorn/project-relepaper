import os
from pathlib import Path

_DEFAULT_PROJECT_PATH = Path(__file__).resolve().parents[2]
PROJECT_PATH = Path(os.getenv("PROJECT_PATH", _DEFAULT_PROJECT_PATH))

_DEFAULT_LOG_DIR = PROJECT_PATH / Path("logs")
LOG_DIR = Path(os.getenv("LOG_DIR", _DEFAULT_LOG_DIR))
