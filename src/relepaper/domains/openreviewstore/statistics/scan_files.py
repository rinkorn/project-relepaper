# %%
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)


# %%
def scan_venue_json_files(directory: Path) -> list[Path]:
    """Поиск файлов venue.json в директории и поддиректориях"""
    directory = Path(directory)
    venue_files = []

    for files in directory.rglob("*.json"):
        if files.name == "venue.json":
            venue_files.append(files)

    return venue_files


def scan_notes_json_files(directory: Path, re_pattern: str = r"^[A-Za-z0-9_-]{8,15}\.json$") -> list[Path]:
    """Поиск файлов <notes>.json в директории и поддиректориях
    Args:
        directory (str or Path): Директория для поиска
        re_pattern (str): Регулярное выражение для поиска файлов

    Returns:
        list[Path]: Список путей к найденным файлам
    """
    directory = Path(directory)
    notes_files = []

    pattern = re.compile(re_pattern)

    for file_path in directory.rglob("*.json"):
        f_pattern = pattern.match if re_pattern.startswith("^") else pattern.search

        if f_pattern(file_path.name):
            notes_files.append(file_path)

    return notes_files


def scan_pdf_files(directory: Path) -> list[Path]:
    """Поиск PDF файлов в директории и поддиректориях
    Args:
        directory (str or Path): Директория для поиска

    Returns:
        list[Path]: Список путей к найденным PDF файлам
    """
    directory = Path(directory)
    pdf_files = []

    for pdf_file in directory.rglob("*/pdf/*.pdf"):
        pdf_files.append(pdf_file)

    return pdf_files


def scan_attachments_files(directory: Path) -> list[Path]:
    """Поиск файлов attachments в директории и поддиректориях
    Args:
        directory (str or Path): Директория для поиска

    Returns:
        list[Path]: Список путей к найденным файлам
    """
    directory = Path(directory)
    attachments_files = []

    for file_path in directory.rglob("*/attachment/*"):
        attachments_files.append(file_path)

    return attachments_files


# %%
if __name__ == "__main__":
    directory = Path("/data/data.sets/openreviewstore/")
    logging.basicConfig(level=logging.INFO)

    venue_files = scan_venue_json_files(directory)
    logger.info(f"Количество обработанных venue.json файлов: {len(venue_files)}")

    notes_files = scan_notes_json_files(directory)
    logger.info(f"Количество загруженных <note>.json файлов: {len(notes_files)}")

    pdf_files = scan_pdf_files(directory)
    logger.info(f"Количество загруженных <pdf>.pdf файлов: {len(pdf_files)}")
