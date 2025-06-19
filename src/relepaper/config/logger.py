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
    name: str = __package__,
    level: int = LOG_LEVEL,
    path: Path = LOG_DIR,
    max_bytes: int = 2000,
    backup_count: int = 5,
    use_stream: bool = True,
    use_file: bool = False,
    stream_level: int = logging.INFO,
    file_level: int = logging.INFO,
    stream_formatter: str = "{asctime} :::: {levelname} :::: {name} :::: {module}:{funcName}:{lineno} :::: {message}",
    file_formatter: str = "{asctime} :::: {levelname} :::: {name} :::: {module}:{funcName}:{lineno} :::: {message}",
    stream_style: str = "{",
    file_style: str = "{",
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if use_stream:
        stream_formatter = logging.Formatter(stream_formatter, style=stream_style)
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(stream_level)
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    if use_file:
        file_formatter = logging.Formatter(file_formatter, style=file_style)
        file_handler_dir = path / Path(f"{__package__}.log")
        file_handler_dir.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_handler_dir,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.debug("The logger has been configured.")
