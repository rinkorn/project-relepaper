# %%
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def download_supplementaries_material_for_note(
    note, venue_path, client, api_version, is_overwrite=False
) -> Dict[str, Any]:
    """Скачивает дополнительные материалы для заметки OpenReview.

    Args:
        note: Объект заметки OpenReview
        venue_path: Путь сохранения
        client: Клиент API OpenReview
        api_version: Версия API
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
        supmat_available_path = None

        if api_version == "v1" and isinstance(note.content, dict):
            supmat_available_path = note.content.get("supplementary_material", "")
        elif api_version == "v2" and isinstance(note.content, dict):
            supmat_available_path = note.content.get("supplementary_material", {}).get("value", "")

        if not supmat_available_path:
            result["error_reason"] = "Путь к дополнительным материалам не указан в заметке"
            return result

        supmat_path = venue_path / Path(supmat_available_path).relative_to("/")
        result["path"] = str(supmat_path)

        # Проверка существования файла
        if supmat_path.is_file():
            result["existed"] = True
            # Нужно ли перезаписать существующий файл
            if not is_overwrite:
                return result
            result["overwritten"] = True

        supmat_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            supmat_binary = client.get_attachment(id=note.id, field_name="supplementary_material")
            with open(supmat_path, "wb") as file:
                file.write(supmat_binary)

            result["success"] = True
            logger.info(f"supmat_downloaded: {supmat_path}")
        except Exception as e:
            result["error"] = True
            result["error_reason"] = f"Ошибка при скачивании доп. материалов: {str(e)}"
            logger.error(f"Ошибка при скачивании дополнительных материалов для заметки {note.id}: {str(e)}")

    except Exception as e:
        result["error"] = True
        result["error_reason"] = f"Общая ошибка: {str(e)}"
        logger.error(f"Общая ошибка при скачивании дополнительных материалов для заметки {note.id}: {str(e)}")

    return result
