import logging
import logging.handlers
import os
import sys
from pathlib import Path

from .constants import LOG_DIR

"""
# NOTSET:0
# DEBUG:10: Detailed information, typically of interest only when diagnosing problems.
# INFO:20: Confirmation that things are working as expected.
# WARNING:30 An indication that something unexpected happened, or indicative of some problem in the near future.
# ERROR:40 Due to a more serious problem, the software has not been able to perform some function.
# CRITITCAL:50 A serious error, indicating that the program itself may be unable to continue running.
"""

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")


def setup_logger(
    name=__package__,
    level=LOG_LEVEL,
    path: Path = LOG_DIR,
    max_bytes: int = 2000,
    backup_count: int = 5,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stream_formatter = logging.Formatter(
        "{asctime} :::: {levelname} :::: {name} :::: {module}:{funcName}:{lineno} :::: {message}",
        style="{",
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(stream_formatter)

    file_formatter = logging.Formatter(
        "{asctime} :::: {levelname} :::: {name} :::: {module}:{funcName}:{lineno} :::: {message}",
        style="{",
    )
    file_handler_dir = path / Path(f"{__package__}.log")
    file_handler_dir.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        file_handler_dir,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.debug("The logger has been configured.")
