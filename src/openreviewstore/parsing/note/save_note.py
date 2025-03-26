# %%
import json
import logging

logger = logging.getLogger(__name__)


def save_note(venue_path, note, additional_data: dict, is_overwrite=False):
    note_path = venue_path / "note" / (note.id + ".json")

    if note_path.is_file() and not is_overwrite:
        return

    note_path.parent.mkdir(parents=True, exist_ok=True)

    note_dict = note.to_json()
    note_dict.update(additional_data)

    with open(note_path, "w", encoding="utf-8") as file:
        json.dump(note_dict, file, indent=2, ensure_ascii=False)

    logger.info(f"note_downloaded: {note_path}")
