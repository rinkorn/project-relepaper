import sys
from pathlib import Path

from loguru import logger

from relepaper.config.constants import LOG_DIR

"""
## The log levels
TRACE:5 | logger.trace() | The lowest level of logging, used for tracing the execution of the program.
DEBUG:10 | logger.debug() | Detailed information, typically of interest only when diagnosing problems.
INFO:20 | logger.info() | Confirmation that things are working as expected.
SUCCESS:25 | logger.success() | The operation was successful.
WARNING:30 | logger.warning() | An indication that something unexpected happened, or indicative of some problem in the near future.
ERROR:40 | logger.error() | Due to a more serious problem, the software has not been able to perform some function.
CRITICAL:50 | logger.critical() | A serious error, indicating that the program itself may be unable to continue running.

## The record dict
elapsed | The time elapsed since the start of the program | See datetime.timedelta
exception | The formatted exception if any, None otherwise | type, value, traceback
extra | The dict of attributes bound by the user (see bind()) | None
file | The file where the logging call was made | name (default), path
function | The function from which the logging call was made | None
level | The severity used to log the message | name (default), no, icon
line | The line number in the source code | None
message | The logged message (not yet formatted) | None
module | The module where the logging call was made | None
name | The __name__ where the logging call was made | None
process | The process in which the logging call was made | name, id (default)
thread | The thread in which the logging call was made | name, id (default)
time | The aware local time when the logging call was made | See datetime.datetime
"""

__all__ = ["setup_logger"]

_DEFAULT_STREAM_LOG_LEVEL = "INFO"
_DEFAULT_FILE_LOG_LEVEL = "DEBUG"
_DEFAULT_STREAM_FORMATTER = (
    "<g>{time:YYYY-MM-DD HH:mm:ss.SSSS}</g> "
    "| <lvl>{level}</lvl> "
    "| <lvl>{extra[classname]}:{function}:{line}</lvl> "
    "| {message}"
)
_DEFAULT_LOG_FORMATTER = (
    "<g>{time:YYYY-MM-DD HH:mm:ss.SSSS}</g> "
    "| <lvl>{level}</lvl> "
    "| <lvl>{name}:{extra[classname]}:{function}:{line}</lvl> "
    "| {message}"
)
_DEFAULT_EXTRA = {"classname": ""}


def setup_logger(
    stream_level: str = _DEFAULT_STREAM_LOG_LEVEL,
    stream_formatter: str = _DEFAULT_STREAM_FORMATTER,
    use_file: bool = False,
    path: Path = LOG_DIR,
    file_level: str = _DEFAULT_FILE_LOG_LEVEL,  # INFO, DEBUG, TRACE, etc.
    file_formatter: str = _DEFAULT_LOG_FORMATTER,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,  # Keep 5 rotated files
    compression: str = "zip",  # Compress the rotated files
    serialize: bool = True,  # Serialize the record dict to a JSON string
    default_extra: dict = _DEFAULT_EXTRA,
) -> None:
    # custom log levels
    # logger.level("FATAL", no=60, color="<red>", icon="!!!")

    logger.configure(extra=default_extra)
    try:
        logger.remove()  # Remove the default handler
        logger.add(
            sys.stderr,
            level=stream_level,
            format=stream_formatter,
            colorize=True,
            backtrace=True,
        )
        logger.info("The default handler has been removed from the logger.")
    except Exception as e:
        logger.error(f"The default handler has not been removed: {e}")
        pass
    if use_file:
        path.mkdir(parents=True, exist_ok=True)
        logger.add(
            path / Path(f"{__package__}.log"),
            level=file_level,
            format=file_formatter,
            rotation=max_bytes,
            retention=backup_count,
            compression=compression,
            serialize=serialize,
        )
        logger.info("The file handler has been added to the logger.")
    else:
        logger.info("The file handler has not been added to the logger.")
    logger.debug("The logger has been configured.")


if __name__ == "__main__":
    stream_formatter = (
        "<g>{time:YYYY-MM-DD HH:mm:ss.SSSS}</g> "
        "| <lvl>{level}</lvl> "
        "| <lvl>{extra[classname]}:{function}:{line}</lvl> "
        "| {message}"
    )
    file_formatter = (
        "<g>{time:YYYY-MM-DD HH:mm:ss.SSSS}</g> "
        "| <lvl>{level}</lvl> "
        "| <lvl>{name}:{extra[classname]}:{function}:{line}</lvl> "
        "| {message}"
    )

    setup_logger(
        stream_level="TRACE",
        stream_formatter=stream_formatter,
        path=LOG_DIR,
        max_bytes=100 * 1024 * 1024,
        backup_count=1,
        use_file=True,
        file_level="TRACE",
        file_formatter=stream_formatter,
        default_extra={"classname": ""},
    )
    logger.trace("trace")
    logger.bind(classname="Hello").trace("trace")
    logger.debug("debug")
    logger.info("info")
    logger.success("success")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
    # logger.fatal("fatal")
