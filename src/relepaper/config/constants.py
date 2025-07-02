import os
from pathlib import Path

_DEFAULT_PROJECT_PATH = Path(__file__).resolve().parents[2]
PROJECT_PATH = Path(os.getenv("PROJECT_PATH", _DEFAULT_PROJECT_PATH))

_DEFAULT_LOG_DIR = PROJECT_PATH / Path("logs")
LOG_DIR = Path(os.getenv("LOG_DIR", _DEFAULT_LOG_DIR))

_DEFAULT_STREAM_LOG_LEVEL = "WARNING"
STREAM_LOG_LEVEL = os.getenv("STREAM_LOG_LEVEL", _DEFAULT_STREAM_LOG_LEVEL)

_DEFAULT_FILE_LOG_LEVEL = "WARNING"
FILE_LOG_LEVEL = os.getenv("FILE_LOG_LEVEL", _DEFAULT_FILE_LOG_LEVEL)

_DEFAULT_LOG_FORMAT = (
    "<g>{time:YYYY-MM-DD HH:mm:SSSS}</g> | <lvl>{level}</lvl> | <lvl>{name}:{function}:{line}</lvl> | {message}"
)
LOG_FORMAT = os.getenv("LOG_FORMAT", _DEFAULT_LOG_FORMAT)
