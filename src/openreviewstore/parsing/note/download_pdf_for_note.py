# %%
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_pdf_for_note(note, venue_path, client, api_version, is_overwrite=False):
    pdf_available_path = None

    if api_version == "v1" and isinstance(note.content, dict):
        pdf_available_path = note.content.get("pdf", "")
    elif api_version == "v2" and isinstance(note.content, dict):
        pdf_available_path = note.content.get("pdf", {}).get("value", "")

    if not pdf_available_path:
        return

    pdf_path = venue_path / Path(pdf_available_path).relative_to("/")

    # нужно ли перезаписать существующий файл
    if pdf_path.is_file() and not is_overwrite:
        return

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_binary = client.get_pdf(id=note.id)
    with open(pdf_path, "wb") as file:
        file.write(pdf_binary)

    logger.info(f"pdf_downloaded: {pdf_path}")
