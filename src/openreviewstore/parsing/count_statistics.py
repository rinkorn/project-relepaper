import argparse
import json
import logging
import re
from pathlib import Path

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
def extract_official_reviews_from_notes(notes_files: list[Path]) -> list[Path]:
    """Извлечение replies из <note>.json
    Args:
        notes_files (list[Path]): Список путей к файлам

    Returns:
        list[Path]: Список путей к файлам с replies
    """
    official_reviews = []
    for note_file in notes_files:
        with note_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if "details" not in data:
                continue
            if "replies" not in data["details"]:
                continue
            if not data["details"]["replies"]:
                continue
            official_reviews.append(note_file)
    return official_reviews


def extract_pdf_name_from_note_json(note_json_file: Path) -> str | None:
    """Извлечение pdf названия из <note>.json
    Args:
        note_json_file (Path): Путь к файлу

    Returns:
        str | None: Название pdf файла или None
    """
    with note_json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if "content" not in data:
            return None
        if "pdf" in data["content"] and isinstance(data["content"]["pdf"], str):
            return data["content"]["pdf"]
        if "pdf" in data["content"] and "value" in data["content"]["pdf"]:
            return data["content"]["pdf"]["value"]
        return None


# %%
def filter_notes_with_replies_field(notes_files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Фильтрация файлов <notes>.json по наличию replies
    Args:
        notes_files (list[Path]): Список путей к файлам

    Returns:
        list[Path]: Список путей к файлам с replies
    """
    notes_with_replies, notes_without_replies = [], []

    for note_file in notes_files:
        if extract_official_reviews_from_notes([note_file]):
            notes_with_replies.append(note_file)
        else:
            notes_without_replies.append(note_file)

    return notes_with_replies, notes_without_replies


def filter_notes_with_pdf_field(notes_files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Фильтрация файлов <notes>.json по наличию pdf
    Args:
        notes_files (list[Path]): Список путей к файлам

    Returns:
        tuple[list[Path], list[Path]]: Список путей к файлам с pdf файлом и список путей к файлам без pdf файла
    """
    notes_with_pdf, notes_without_pdf = [], []
    for note_file in notes_files:
        pdf_name = extract_pdf_name_from_note_json(note_file)
        if pdf_name:
            notes_with_pdf.append(note_file)
        else:
            notes_without_pdf.append(note_file)
    return notes_with_pdf, notes_without_pdf


def filter_notes_with_pdf_field_and_pdf_files(
    notes_files: list[Path], pdf_files: list[Path]
) -> tuple[list[Path], list[Path]]:
    """Фильтрация файлов <notes>.json по наличию pdf и сопоставление с pdf файлами в директории
    Args:
        notes_files (list[Path]): Список путей к файлам
        pdf_files (list[Path]): Список путей к файлам

    Returns:
        tuple[list[Path], list[Path]]: Список путей к файлам с pdf файлом и список путей к файлам без pdf файла
    """

    notes_files_filtered, notes_files_not_filtered = [], []

    def compare_pdf_name_with_pdf_files(note_file: Path):
        pdf_name = extract_pdf_name_from_note_json(note_file).split("/")[-1]
        if pdf_name in [pdf_file.name for pdf_file in pdf_files]:
            notes_files_filtered.append(note_file)
        else:
            notes_files_not_filtered.append(note_file)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(compare_pdf_name_with_pdf_files, notes_files)

    return notes_files_filtered, notes_files_not_filtered


# %%
def main(store_path: Path):
    """Основная функция для подсчета файлов"""

    # scan files
    venue_files = scan_venue_json_files(store_path)
    notes_files = scan_notes_json_files(store_path)
    pdf_files = scan_pdf_files(store_path)
    # filter notes files
    notes_with_replies, _ = filter_notes_with_replies_field(notes_files)
    notes_with_pdf, _ = filter_notes_with_pdf_field(notes_files)
    notes_with_replies_and_pdf, _ = filter_notes_with_pdf_field(notes_with_replies)
    notes_with_replies_and_pdf_and_pdf_files, _ = filter_notes_with_pdf_field_and_pdf_files(
        notes_with_replies_and_pdf, pdf_files
    )

    print(f"Количество обработанных venue.json файлов: {len(venue_files)}")
    print(f"Количество загруженных <note>.json файлов: {len(notes_files)}")
    print(f"Количество загруженных <pdf>.pdf файлов: {len(pdf_files)}")
    print(f"Количество <note>.json, с заполненным полем <pdf>: {len(notes_with_pdf)}")
    print(f"Количество <note>.json, с заполненным полем <replies>: {len(notes_with_replies)}")
    print(f"Количество <note>.json, с заполненными полями <pdf> и <replies>: {len(notes_with_replies_and_pdf)}")
    print(
        f"Количество <note>.json, с заполненными полями <pdf> и <replies> и существующими PDF файлами: {len(notes_with_replies_and_pdf_and_pdf_files)}"
    )
    return (
        venue_files,
        notes_files,
        pdf_files,
        notes_with_replies,
        notes_with_pdf,
        notes_with_replies_and_pdf,
        notes_with_replies_and_pdf_and_pdf_files,
    )


def cli():
    parser = argparse.ArgumentParser(
        description="Подсчет .pdf и .json файлов с определенными условиями.",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Путь к директории для поиска файлов",
    )
    args = parser.parse_args()
    store_path = args.directory
    main(store_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # store_path = Path("/home/rinkorn/space/prog/python/sber/project-openreviewstore/data/store/")
    store_path = Path("/data/data.sets/openreviewstore/")
    p_list = main(store_path)
    venue_files = p_list[0]
    notes_files = p_list[1]
    pdf_files = p_list[2]
    notes_with_replies = p_list[3]
    notes_with_pdf = p_list[4]
    notes_with_replies_and_pdf = p_list[5]
    notes_with_replies_and_pdf_and_pdf_files = p_list[6]
