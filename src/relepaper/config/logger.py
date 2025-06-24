import sys
from pathlib import Path

from loguru import logger

from relepaper.config.constants import (
    FILE_LOG_LEVEL,
    LOG_DIR,
    LOG_FORMAT,
    STREAM_LOG_LEVEL,
)

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


def setup_logger(
    stream_level: str = STREAM_LOG_LEVEL,
    stream_formatter: str = LOG_FORMAT,
    use_file: bool = False,
    path: Path = LOG_DIR,
    file_level: str = FILE_LOG_LEVEL,  # INFO, DEBUG, TRACE, etc.
    file_formatter: str = LOG_FORMAT,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,  # Keep 5 rotated files
    compression: str = "zip",  # Compress the rotated files
    serialize: bool = True,  # Serialize the record dict to a JSON string
) -> None:
    # custom log levels
    # logger.level("FATAL", no=60, color="<red>", icon="!!!")

    try:
        logger.remove(0)  # Remove the default handler
        logger.add(
            sys.stderr,
            level=stream_level,
            format=stream_formatter,
            colorize=True,
            backtrace=True,
        )
    except Exception:
        # logger.error(f"Error removing the default handler: {e}")
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

    logger.debug("The logger has been configured.")


if __name__ == "__main__":
    setup_logger(
        stream_level="TRACE",
        path=LOG_DIR,
        max_bytes=100 * 1024 * 1024,
        backup_count=1,
        use_file=True,
    )
    logger.trace("trace")
    logger.debug("debug")
    logger.info("info")
    logger.success("success")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
    # logger.fatal("fatal")
