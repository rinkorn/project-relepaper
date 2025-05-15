# %%
import logging
from pathlib import Path

from relepaper.store.openreviewnet.statistics.extract_from_note import (
    extract_pdf_name_from_note_json,
    extract_replies_from_note_json,
    extract_responses_from_note_json,
)

logger = logging.getLogger(__name__)


# %%
def filter_by_pdf_field(notes_files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Фильтрация файлов <notes>.json по наличию pdf
    Args:
        notes_files (list[Path]): Список путей к файлам

    Returns:
        tuple[list[Path], list[Path]]: Список путей к файлам с pdf файлом и список путей к файлам без pdf файла
    """
    notes_with_pdf_field, notes_without_pdf_field = [], []
    for note_file in notes_files:
        pdf_field = extract_pdf_name_from_note_json(note_file)
        if pdf_field:
            notes_with_pdf_field.append(note_file)
        else:
            notes_without_pdf_field.append(note_file)
    return notes_with_pdf_field, notes_without_pdf_field


def filter_by_existing_pdf(notes_files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Фильтрация файлов <note>.json по наличию pdf и сопоставление с pdf файлами в директории
    Args:
        notes_files (list[Path]): Список путей к файлам

    Returns:
        tuple[list[Path], list[Path]]: Список путей к файлам с pdf файлом и список путей к файлам без pdf файла
    """

    notes_with_existing_pdf, notes_not_existing_pdf = [], []

    for note_file in notes_files:
        pdf_field = extract_pdf_name_from_note_json(note_file)
        if not pdf_field:
            notes_not_existing_pdf.append(note_file)
            continue

        # reconstruct pdf path
        pdf_path = note_file.parents[1] / Path(pdf_field).relative_to("/")

        # check pdf path existence
        if pdf_path.is_file():
            notes_with_existing_pdf.append(note_file)
        else:
            notes_not_existing_pdf.append(note_file)

    return notes_with_existing_pdf, notes_not_existing_pdf


def filter_by_replies_field(notes_files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Фильтрация файлов <notes>.json по наличию replies
    Args:
        notes_files (list[Path]): Список путей к файлам

    Returns:
        list[Path]: Список путей к файлам с replies
    """
    notes_with_replies_field, notes_without_replies_field = [], []

    for note_file in notes_files:
        if len(extract_replies_from_note_json(note_file)) > 0:
            notes_with_replies_field.append(note_file)
        else:
            notes_without_replies_field.append(note_file)

    return notes_with_replies_field, notes_without_replies_field


def filter_by_responses_field(notes_files: list[Path], invitation_endswith: str) -> tuple[list[Path], list[Path]]:
    """Фильтрация файлов <notes>.json по наличию определенного вида responses
    Args:
        notes_files (list[Path]): Список путей к файлам
        invitation_endswith (str): Конец строки в invitation

    Returns:
        list[Path]: Список путей к файлам с определенным видом responses
    """
    notes_with_responses_field, notes_without_responses_field = [], []

    for note_file in notes_files:
        responses = extract_responses_from_note_json(note_file, invitation_endswith)
        if len(responses) > 0:
            notes_with_responses_field.append(note_file)
        else:
            notes_without_responses_field.append(note_file)

    return notes_with_responses_field, notes_without_responses_field


# %%
if __name__ == "__main__":
    from relepaper.store.openreviewnet.statistics.scan_files import (
        scan_notes_json_files,
        scan_pdf_files,
        scan_venue_json_files,
    )

    logging.basicConfig(level=logging.INFO)

    # store_path = Path("/home/rinkorn/space/prog/python/sber/project-openreviewstore/data/store/")
    store_path = Path("/data/data.sets/openreviewstore/")

    # scanning
    venue_files = scan_venue_json_files(store_path)
    print(f"Количество обработанных venue.json файлов: {len(venue_files)}")

    notes_files = scan_notes_json_files(store_path)
    print(f"Количество загруженных <note>.json файлов: {len(notes_files)}")

    pdf_files = scan_pdf_files(store_path)
    print(f"Количество загруженных <pdf>.pdf файлов: {len(pdf_files)}")

    # filtering
    notes, _ = filter_by_pdf_field(notes_files)
    print(
        "Количество <note>.json, с:\n",
        " * заполненным полем <pdf>\n",
        len(notes),
    )

    notes, _ = filter_by_replies_field(notes_files)
    print(
        "Количество <note>.json, с:\n",
        " * заполненным полем <pdf>\n",
        " * заполненным полем <replies>\n",
        len(notes),
    )

    notes, _ = filter_by_replies_field(notes_files)
    notes, _ = filter_by_pdf_field(notes)
    print(
        "Количество <note>.json, с:\n",
        " * заполненными полями <pdf>\n",
        " * заполненными полями <replies>\n",
        len(notes),
    )

    notes, _ = filter_by_replies_field(notes_files)
    notes, _ = filter_by_pdf_field(notes)
    notes, _ = filter_by_existing_pdf(notes)
    print(
        "Количество <note>.json, с:\n",
        " * заполненными полями <pdf>\n",
        " * заполненными полями <replies>\n",
        " * существующими PDF файлами\n",
        len(notes),
    )

    notes, _ = filter_by_replies_field(notes_files)
    notes, _ = filter_by_pdf_field(notes)
    notes, _ = filter_by_existing_pdf(notes)
    notes, _ = filter_by_responses_field(notes, "Official_Review")
    print(
        "Количество <note>.json, с:\n",
        " * заполненными полями <pdf>\n",
        " * заполненными полями <replies>\n",
        " * существующими PDF файлами\n",
        " * присутствующими Official_Review\n",
        len(notes),
    )

    notes, _ = filter_by_replies_field(notes_files)
    notes, _ = filter_by_pdf_field(notes)
    notes, _ = filter_by_existing_pdf(notes)
    notes, _ = filter_by_responses_field(notes, "Official_Review")
    notes, _ = filter_by_responses_field(notes, "Decision")
    print(
        "Количество <note>.json, с:\n",
        " * заполненными полями <pdf>\n",
        " * заполненными полями <replies>\n",
        " * существующими PDF файлами\n",
        " * присутствующими Official_Review\n",
        " * присутствующими Decision\n",
        len(notes),
    )
