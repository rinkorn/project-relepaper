# %%
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def save_note(venue_path, note, additional_data: dict, is_overwrite=False) -> Dict[str, Any]:
    """Сохраняет заметку OpenReview в JSON файл.

    Args:
        venue_path: Путь сохранения
        note: Объект заметки OpenReview
        additional_data: Дополнительные данные для включения в JSON
        is_overwrite: Флаг перезаписи существующих файлов

    Returns:
        Dict[str, Any]: Словарь с результатами операции, содержащий:
            success (bool): Успешно ли выполнена операция
            existed (bool): Существовал ли файл ранее
            overwritten (bool): Был ли перезаписан файл
            path (str): Путь к файлу (если есть)
            error (bool): Была ли ошибка
            error_reason (str): Причина ошибки (если есть)
    """
    result = {
        "success": False,
        "existed": False,
        "overwritten": False,
        "path": None,
        "error": False,
        "error_reason": None,
    }

    try:
        note_path = venue_path / "note" / (note.id + ".json")
        result["path"] = str(note_path)

        # Check if the file exists
        if note_path.is_file():
            result["existed"] = True
            # Need to overwrite the existing file
            if not is_overwrite:
                return result
            result["overwritten"] = True

        note_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            note_dict = note.to_json()
            note_dict.update(additional_data)

            with open(note_path, "w", encoding="utf-8") as file:
                json.dump(
                    note_dict,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            result["success"] = True
            logger.info(f"note_saved: {note_path}")
        except Exception as e:
            result["error"] = True
            result["error_reason"] = f"Error saving note: {str(e)}"
            logger.error(f"Error saving note {note.id}: {str(e)}")

    except Exception as e:
        result["error"] = True
        result["error_reason"] = f"General error: {str(e)}"
        logger.error(f"General error saving note {note.id}: {str(e)}")

    return result
