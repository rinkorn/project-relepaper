# %%
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def download_pdf_for_note(note, venue_path, client, api_version, is_overwrite=False) -> Dict[str, Any]:
    """Скачивает PDF для заметки OpenReview.

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
        pdf_available_path = None

        if api_version == "v1" and isinstance(note.content, dict):
            pdf_available_path = note.content.get("pdf", "")
        elif api_version == "v2" and isinstance(note.content, dict):
            pdf_available_path = note.content.get("pdf", {}).get("value", "")

        if not pdf_available_path:
            result["error"] = True
            result["error_reason"] = "PDF путь не указан в заметке"
            return result

        pdf_path = venue_path / Path(pdf_available_path).relative_to("/")
        result["path"] = str(pdf_path)

        # Проверка существования файла
        if pdf_path.is_file():
            result["existed"] = True
            # Нужно ли перезаписать существующий файл
            if not is_overwrite:
                return result
            result["overwritten"] = True

        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            pdf_binary = client.get_pdf(id=note.id)
            with open(pdf_path, "wb") as file:
                file.write(pdf_binary)

            result["success"] = True
            logger.info(f"pdf_downloaded: {pdf_path}")
        except Exception as e:
            result["error"] = True
            result["error_reason"] = f"Ошибка при скачивании PDF: {str(e)}"
            logger.error(f"Ошибка при скачивании PDF для заметки {note.id}: {str(e)}")

    except Exception as e:
        result["error"] = True
        result["error_reason"] = f"Общая ошибка: {str(e)}"
        logger.error(f"Общая ошибка при скачивании PDF для заметки {note.id}: {str(e)}")

    return result
