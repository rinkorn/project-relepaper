import os
from pathlib import Path

_DEFAULT_PROJECT_PATH = Path(__file__).resolve().parents[2]
PROJECT_PATH = Path(os.getenv("PROJECT_PATH", _DEFAULT_PROJECT_PATH))
