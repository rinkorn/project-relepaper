# %%
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_supplementary_material_for_note(note, venue_path, client, api_version, is_overwrite=False):
    supmat_available_path = None

    if api_version == "v1" and isinstance(note.content, dict):
        supmat_available_path = note.content.get("supplementary_material", "")
    elif api_version == "v2" and isinstance(note.content, dict):
        supmat_available_path = note.content.get("supplementary_material", {}).get("value", "")

    if not supmat_available_path:
        return

    supmat_path = venue_path / Path(supmat_available_path).relative_to("/")

    # нужно ли перезаписать существующий файл
    if supmat_path.is_file() and not is_overwrite:
        return

    supmat_path.parent.mkdir(parents=True, exist_ok=True)

    supmat_binary = client.get_attachment(id=note.id, field_name="supplementary_material")
    with open(supmat_path, "wb") as file:
        file.write(supmat_binary)

    logger.info(f"supmat_downloaded: {supmat_path}")
