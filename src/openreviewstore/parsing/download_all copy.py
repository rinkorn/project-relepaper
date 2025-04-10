# %%
import argparse
import logging
import time
import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict, Any

from openreviewstore.constants import PROJECT_PATH
from openreviewstore.parsing import (
    save_venue,
    save_note,
    OpenReviewClients,
    download_pdf_for_note,
    download_supplementary_material_for_note,
    get_accepted_submissions_for_double_blind_venues_apiv1,
    get_accepted_submissions_for_single_blind_venues_apiv1,
    get_accepted_submissions_of_venue_apiv2,
    get_active_submissions_for_a_double_blind_venue_apiv1,
    get_active_submissions_under_review_of_venue_apiv2,
    get_all_submissions_for_a_double_blind_venue_apiv1,
    get_all_the_submissions_notes_of_venue_apiv2,
    get_all_venues_name,
    get_desk_rejected_submissions_of_venue_apiv2,
    get_simple_all_the_submissions_notes_of_venue_apiv2,
    get_withdrawn_submissions_of_venue_apiv2,
    identify_client_api_version_for_venue,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки.

    Returns:
        argparse.Namespace: Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(description="Скрипт для скачивания данных с OpenReview")
    parser.add_argument(
        "--store-path",
        type=str,
        default=str(PROJECT_PATH / "data/store/"),
        help="Путь для сохранения данных (по умолчанию: data/store/)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Максимальное количество параллельных обработчиков (по умолчанию: 5)",
    )
    parser.add_argument(
        "--venue-id",
        type=str,
        nargs="+",
        default=None,
        help="ID мероприятий для обработки (можно указать несколько через пробел)",
    )
    parser.add_argument(
        "--overwrite-notes",
        action="store_true",
        help="Перезаписать существующие файлы заметок",
    )
    parser.add_argument(
        "--overwrite-pdfs",
        action="store_true",
        help="Перезаписать существующие PDF-файлы",
    )
    parser.add_argument(
        "--overwrite-supplementary",
        action="store_true",
        help="Перезаписать существующие дополнительные материалы",
    )
    parser.add_argument(
        "--overwrite-venue",
        action="store_true",
        help="Перезаписать существующие файлы информации о мероприятии",
    )
    parser.add_argument(
        "--overwrite-all",
        action="store_true",
        help="Перезаписать все существующие файлы",
    )
    parser.add_argument("--debug", action="store_true", help="Включить отладочные сообщения")
    parser.add_argument("--no-report", action="store_true", help="Не генерировать HTML-отчёт со статистикой")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Только сгенерировать отчёт на основе имеющихся данных без скачивания",
    )
    return parser.parse_args()


def process_note(
    note: Any,
    venue_path: Path,
    client: Any,
    api_version: str,
    overwrite_notes: bool = False,
    overwrite_pdfs: bool = False,
    overwrite_supplementary: bool = False,
) -> Dict[str, Any]:
    """Обрабатывает отдельную заметку (note) и скачивает связанные с ней данные.

    Args:
        note: Объект заметки OpenReview
        venue_path: Путь для сохранения данных
        client: Клиент API OpenReview
        api_version: Версия API
        overwrite_notes: Флаг перезаписи файлов заметок
        overwrite_pdfs: Флаг перезаписи PDF-файлов
        overwrite_supplementary: Флаг перезаписи дополнительных материалов

    Returns:
        Dict[str, Any]: Словарь со статистикой обработки заметки
    """
    result = {
        "pdf_success": False,  # Успешно скачан PDF
        "pdf_existed": False,  # PDF уже существовал
        "pdf_overwritten": False,  # PDF был перезаписан
        "pdf_error": False,  # Ошибка при скачивании PDF
        "supp_success": False,  # Успешно скачаны дополнительные материалы
        "supp_existed": False,  # Дополнительные материалы уже существовали
        "supp_overwritten": False,  # Дополнительные материалы были перезаписаны
        "supp_error": False,  # Ошибка при скачивании дополнительных материалов
        "note_success": False,  # Успешно сохранена заметка
        "note_existed": False,  # Заметка уже существовала
        "note_overwritten": False,  # Заметка была перезаписана
        "note_error": False,  # Ошибка при сохранении заметки
    }

    try:
        # Сохраняем заметку
        try:
            note_result = save_note(
                venue_path,
                note,
                additional_data={"details": note.details, "api_version": api_version},
                is_overwrite=overwrite_notes,
            )

            if isinstance(note_result, dict):
                result["note_success"] = note_result.get("success", False)
                result["note_existed"] = note_result.get("existed", False)
                result["note_overwritten"] = note_result.get("overwritten", False)
            else:
                result["note_success"] = bool(note_result)
        except Exception as e:
            logger.error(f"Ошибка при сохранении заметки {note.id}: {str(e)}")
            result["note_error"] = True

        # Скачивание PDF
        try:
            pdf_result = download_pdf_for_note(
                note,
                venue_path,
                client,
                api_version,
                is_overwrite=overwrite_pdfs,
            )

            if isinstance(pdf_result, dict):
                result["pdf_success"] = pdf_result.get("success", False)
                result["pdf_existed"] = pdf_result.get("existed", False)
                result["pdf_overwritten"] = pdf_result.get("overwritten", False)
            else:
                result["pdf_success"] = bool(pdf_result)
        except Exception as e:
            logger.error(f"Ошибка при скачивании PDF для заметки {note.id}: {str(e)}")
            result["pdf_error"] = True

        # Скачивание дополнительных материалов
        try:
            supp_result = download_supplementary_material_for_note(
                note,
                venue_path,
                client,
                api_version,
                is_overwrite=overwrite_supplementary,
            )

            if isinstance(supp_result, dict):
                result["supp_success"] = supp_result.get("success", False)
                result["supp_existed"] = supp_result.get("existed", False)
                result["supp_overwritten"] = supp_result.get("overwritten", False)
            else:
                result["supp_success"] = bool(supp_result)
        except Exception as e:
            logger.error(f"Ошибка при скачивании дополнительных материалов для заметки {note.id}: {str(e)}")
            result["supp_error"] = True

    except Exception as e:
        logger.error(f"Общая ошибка при обработке заметки {note.id}: {str(e)}")

    return result


def get_notes_for_venue(client: Any, venue_id: str, api_version: str) -> Optional[List[Any]]:
    """Получает список заметок для указанного мероприятия, последовательно пробуя
    различные методы API в зависимости от версии.

    Args:
        client: Клиент API OpenReview
        venue_id: ID мероприятия
        api_version: Версия API

    Returns:
        Optional[List[Any]]: Список заметок или None в случае ошибки
    """
    notes = []

    if api_version == "v1":
        retrieval_methods = [
            lambda: get_active_submissions_for_a_double_blind_venue_apiv1(client, venue_id),
            lambda: get_all_submissions_for_a_double_blind_venue_apiv1(client, venue_id),
            lambda: get_accepted_submissions_for_double_blind_venues_apiv1(client, venue_id),
            lambda: get_accepted_submissions_for_single_blind_venues_apiv1(client, venue_id),
        ]
    elif api_version == "v2":
        retrieval_methods = [
            lambda: get_all_the_submissions_notes_of_venue_apiv2(client, venue_id),
            lambda: get_simple_all_the_submissions_notes_of_venue_apiv2(client, venue_id),
            lambda: get_accepted_submissions_of_venue_apiv2(client, venue_id),
            lambda: get_active_submissions_under_review_of_venue_apiv2(client, venue_id),
            lambda: get_withdrawn_submissions_of_venue_apiv2(client, venue_id),
            lambda: get_desk_rejected_submissions_of_venue_apiv2(client, venue_id),
        ]
    else:
        logger.error(f"Неподдерживаемая версия API: {api_version}")
        return None

    for method in retrieval_methods:
        try:
            result = method()
            if result:
                notes = result
                break
        except Exception as e:
            logger.debug(f"Ошибка при получении заметок: {str(e)}")

    return notes


def process_venue(
    venue_id: str,
    clients: OpenReviewClients,
    store_path: Path,
    max_workers: int,
    overwrite_notes: bool = False,
    overwrite_pdfs: bool = False,
    overwrite_supplementary: bool = False,
    overwrite_venue: bool = False,
    report_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Обрабатывает отдельное мероприятие.

    Args:
        venue_id: ID мероприятия
        clients: Объект клиентов OpenReview
        store_path: Базовый путь для сохранения данных
        max_workers: Максимальное количество параллельных обработчиков
        overwrite_notes: Флаг перезаписи файлов заметок
        overwrite_pdfs: Флаг перезаписи PDF-файлов
        overwrite_supplementary: Флаг перезаписи дополнительных материалов
        overwrite_venue: Флаг перезаписи файлов информации о мероприятии
        report_file: Путь к файлу отчета

    Returns:
        Dict[str, Any]: Результаты обработки мероприятия
    """
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    time.sleep(1)  # Небольшая задержка для предотвращения блокировки API
    result = {
        "venue_id": venue_id,
        "status": "error",
        "api_version": None,
        "notes_count": 0,
        "pdf_count": 0,
        "supplementary_count": 0,
        "timestamp": timestamp,
        "processing_time": 0,
        # Дополнительная статистика
        "notes_existed": 0,
        "notes_overwritten": 0,
        "notes_error": 0,
        "pdfs_existed": 0,
        "pdfs_overwritten": 0,
        "pdfs_error": 0,
        "supps_existed": 0,
        "supps_overwritten": 0,
        "supps_error": 0,
    }

    api_version = identify_client_api_version_for_venue(clients, venue_id)
    result["api_version"] = api_version

    if api_version is None:
        logger.error(f"Не удалось определить версию API для мероприятия: {venue_id}")
        result["processing_time"] = round(time.time() - start_time, 2)
        _write_report_line(report_file, result)
        return result

    client = clients.get_client(api_version=api_version)
    logger.info(f"Обработка мероприятия: {venue_id} (API: {api_version})")

    try:
        venue_group = client.get_group(venue_id)
        venue_path = store_path / venue_id

        # Сохранение информации о мероприятии
        save_venue(
            venue_group,
            venue_path / "venue.json",
            additional_data={"api_version": api_version},
            is_overwrite=overwrite_venue,
        )

        notes = get_notes_for_venue(client, venue_id, api_version)

        if notes is None:
            logger.error(f"Ошибка при получении заметок для мероприятия: {venue_id}")
            result["processing_time"] = round(time.time() - start_time, 2)
            _write_report_line(report_file, result)
            return result

        if not notes:
            logger.warning(f"Не найдено заметок для мероприятия: {venue_id}")
            result["status"] = "empty"
            result["processing_time"] = round(time.time() - start_time, 2)
            _write_report_line(report_file, result)
            return result

        result["notes_count"] = len(notes)
        logger.info(f"Найдено {len(notes)} заметок для мероприятия: {venue_id}")

        # Параллельная обработка заметок с использованием ThreadPoolExecutor
        pdf_count = 0
        supplementary_count = 0

        # Статистика
        notes_existed = 0
        notes_overwritten = 0
        notes_error = 0
        pdfs_existed = 0
        pdfs_overwritten = 0
        pdfs_error = 0
        supps_existed = 0
        supps_overwritten = 0
        supps_error = 0

        from concurrent.futures import as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_note = {}

            for i_note, note in enumerate(notes):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Заметка {i_note + 1}/{len(notes)}: id={note.id}, number={getattr(note, 'number', 'N/A')}"
                    )
                    logger.debug(f"PDF: {note.content.get('pdf')}")
                    logger.debug(f"Доп. материалы: {note.content.get('supplementary_material')}")
                    logger.debug(f"Ответы: {note.details.get('replies')}")

                future = executor.submit(
                    process_note,
                    note,
                    venue_path,
                    client,
                    api_version,
                    overwrite_notes,
                    overwrite_pdfs,
                    overwrite_supplementary,
                )
                future_to_note[future] = note

            # Обработка результатов выполнения
            for future in as_completed(future_to_note):
                note_result = future.result()

                # Обновление статистики по PDF
                if note_result["pdf_success"]:
                    pdf_count += 1
                if note_result["pdf_existed"]:
                    pdfs_existed += 1
                if note_result["pdf_overwritten"]:
                    pdfs_overwritten += 1
                if note_result["pdf_error"]:
                    pdfs_error += 1

                # Обновление статистики по дополнительным материалам
                if note_result["supp_success"]:
                    supplementary_count += 1
                if note_result["supp_existed"]:
                    supps_existed += 1
                if note_result["supp_overwritten"]:
                    supps_overwritten += 1
                if note_result["supp_error"]:
                    supps_error += 1

                # Обновление статистики по заметкам
                if note_result["note_existed"]:
                    notes_existed += 1
                if note_result["note_overwritten"]:
                    notes_overwritten += 1
                if note_result["note_error"]:
                    notes_error += 1

        # Обновление результатов
        result["pdf_count"] = pdf_count
        result["supplementary_count"] = supplementary_count
        result["notes_existed"] = notes_existed
        result["notes_overwritten"] = notes_overwritten
        result["notes_error"] = notes_error
        result["pdfs_existed"] = pdfs_existed
        result["pdfs_overwritten"] = pdfs_overwritten
        result["pdfs_error"] = pdfs_error
        result["supps_existed"] = supps_existed
        result["supps_overwritten"] = supps_overwritten
        result["supps_error"] = supps_error

        result["status"] = "success"
        result["processing_time"] = round(time.time() - start_time, 2)

        _write_report_line(report_file, result)
        return result

    except Exception as e:
        logger.error(f"Ошибка при обработке мероприятия {venue_id}: {str(e)}")
        result["processing_time"] = round(time.time() - start_time, 2)
        _write_report_line(report_file, result)
        return result


def _write_report_line(report_file: Optional[Path], result: Dict[str, Any]) -> None:
    """Записывает строку в файл отчета.

    Args:
        report_file: Путь к файлу отчета
        result: Словарь с результатами обработки мероприятия
    """
    if report_file is None:
        return

    try:
        # Создаем заголовок при необходимости
        if not report_file.exists() or report_file.stat().st_size == 0:
            header = (
                "timestamp\tvenue_id\tstatus\tapi_version\t"
                "notes_count\tpdf_count\tsupplementary_count\t"
                "notes_existed\tnotes_overwritten\tnotes_error\t"
                "pdfs_existed\tpdfs_overwritten\tpdfs_error\t"
                "supps_existed\tsupps_overwritten\tsupps_error\t"
                "processing_time\n"
            )
            with open(report_file, "w") as f:
                f.write(header)

        # Записываем строку с данными
        line = (
            f"{result['timestamp']}\t"
            f"{result['venue_id']}\t"
            f"{result['status']}\t"
            f"{result['api_version'] or 'unknown'}\t"
            f"{result['notes_count']}\t"
            f"{result['pdf_count']}\t"
            f"{result['supplementary_count']}\t"
            f"{result.get('notes_existed', 0)}\t"
            f"{result.get('notes_overwritten', 0)}\t"
            f"{result.get('notes_error', 0)}\t"
            f"{result.get('pdfs_existed', 0)}\t"
            f"{result.get('pdfs_overwritten', 0)}\t"
            f"{result.get('pdfs_error', 0)}\t"
            f"{result.get('supps_existed', 0)}\t"
            f"{result.get('supps_overwritten', 0)}\t"
            f"{result.get('supps_error', 0)}\t"
            f"{result['processing_time']}s\n"
        )

        with open(report_file, "a") as f:
            f.write(line)

    except Exception as e:
        logger.error(f"Ошибка при записи в файл отчета: {str(e)}")


def generate_statistics_report(
    results: Dict[str, List[Dict[str, Any]]], store_path: Path, report_filename: str = "statistics_report.html"
) -> Path:
    """Генерирует наглядный HTML-отчёт со статистикой загрузки данных.

    Args:
        results: Словарь с результатами обработки мероприятий
                (ключи: "success", "empty", "error")
        store_path: Базовый путь хранилища данных
        report_filename: Имя файла отчёта

    Returns:
        Path: Путь к созданному файлу отчёта
    """
    # Подготовка данных для отчёта
    total_venues = len(results["success"]) + len(results["empty"]) + len(results["error"])
    success_venues = len(results["success"])
    empty_venues = len(results["empty"])
    error_venues = len(results["error"])

    # Подсчёт общей статистики
    total_notes = sum(r.get("notes_count", 0) for r in results["success"])
    total_pdfs = sum(r.get("pdf_count", 0) for r in results["success"])
    total_supp = sum(r.get("supplementary_count", 0) for r in results["success"])

    # Подсчёт статистики по статусам файлов
    total_notes_existed = sum(r.get("notes_existed", 0) for r in results["success"])
    total_notes_overwritten = sum(r.get("notes_overwritten", 0) for r in results["success"])
    total_notes_error = sum(r.get("notes_error", 0) for r in results["success"])

    total_pdfs_existed = sum(r.get("pdfs_existed", 0) for r in results["success"])
    total_pdfs_overwritten = sum(r.get("pdfs_overwritten", 0) for r in results["success"])
    total_pdfs_error = sum(r.get("pdfs_error", 0) for r in results["success"])

    total_supp_existed = sum(r.get("supps_existed", 0) for r in results["success"])
    total_supp_overwritten = sum(r.get("supps_overwritten", 0) for r in results["success"])
    total_supp_error = sum(r.get("supps_error", 0) for r in results["success"])

    # Расчёт процентов успешности
    success_rate = (success_venues / total_venues) * 100 if total_venues > 0 else 0
    pdf_success_rate = (total_pdfs / total_notes) * 100 if total_notes > 0 else 0
    supp_success_rate = (total_supp / total_notes) * 100 if total_notes > 0 else 0

    # Сортировка успешных мероприятий по количеству нот
    top_venues = sorted(results["success"], key=lambda x: x.get("notes_count", 0), reverse=True)[:10]  # Топ-10

    # Время выполнения
    total_time = sum(r.get("processing_time", 0) for r in results["success"] + results["empty"] + results["error"])
    avg_time_per_venue = total_time / total_venues if total_venues > 0 else 0

    # Генерация HTML-отчёта
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>OpenReview Data Download Report</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            h1, h2, h3, h4 {{
                color: #2c3e50;
            }}
            
            .header {{
                background-color: #3498db;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
                text-align: center;
            }}
            
            .summary-cards {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .card {{
                flex: 1;
                min-width: 200px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 20px;
                text-align: center;
            }}
            
            .card h3 {{
                margin-top: 0;
                color: #7f8c8d;
                font-size: 0.9em;
                text-transform: uppercase;
            }}
            
            .card .value {{
                font-size: 2.5em;
                font-weight: bold;
                color: #2c3e50;
            }}
            
            .card .subvalue {{
                font-size: 1em;
                color: #7f8c8d;
            }}
            
            .success {{
                border-top: 4px solid #2ecc71;
            }}
            
            .warning {{
                border-top: 4px solid #f39c12;
            }}
            
            .error {{
                border-top: 4px solid #e74c3c;
            }}
            
            .info {{
                border-top: 4px solid #3498db;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            
            table th, table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            
            table th {{
                background-color: #f8f9fa;
                font-weight: bold;
                color: #2c3e50;
            }}
            
            table tr:hover {{
                background-color: #f1f1f1;
            }}
            
            .progress-bar {{
                background-color: #ecf0f1;
                border-radius: 13px;
                padding: 3px;
                margin-top: 5px;
            }}
            
            .progress {{
                background-color: #2ecc71;
                height: 10px;
                border-radius: 10px;
                width: 0%;
                transition: width 1s ease-in-out;
            }}
            
            .timestamp {{
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 50px;
            }}
            
            .section {{
                margin-bottom: 40px;
            }}
            
            .column-container {{
                display: flex;
                gap: 20px;
            }}
            
            .column {{
                flex: 1;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Отчёт о загрузке данных с OpenReview</h1>
            <p>Сгенерирован: {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Общая информация</h2>
            <div class="summary-cards">
                <div class="card success">
                    <h3>Успешно обработано</h3>
                    <div class="value">{success_venues}</div>
                    <div class="subvalue">из {total_venues} мероприятий</div>
                    <div class="progress-bar">
                        <div class="progress" style="width: {success_rate:.1f}%"></div>
                    </div>
                </div>
                
                <div class="card warning">
                    <h3>Пустых мероприятий</h3>
                    <div class="value">{empty_venues}</div>
                    <div class="subvalue">мероприятий без данных</div>
                </div>
                
                <div class="card error">
                    <h3>Ошибок</h3>
                    <div class="value">{error_venues}</div>
                    <div class="subvalue">не удалось обработать</div>
                </div>
                
                <div class="card info">
                    <h3>Общее время</h3>
                    <div class="value">{total_time:.1f}s</div>
                    <div class="subvalue">~{avg_time_per_venue:.1f}s на мероприятие</div>
                </div>
            </div>
        </div>
        
        <div class="section column-container">
            <div class="column">
                <h2>Статистика загрузок</h2>
                <div class="summary-cards">
                    <div class="card info">
                        <h3>Заметки</h3>
                        <div class="value">{total_notes}</div>
                        <div class="subvalue">
                            {total_notes_existed} существовало, 
                            {total_notes_overwritten} перезаписано, 
                            {total_notes_error} ошибок
                        </div>
                    </div>
                    
                    <div class="card success">
                        <h3>PDF-файлы</h3>
                        <div class="value">{total_pdfs}</div>
                        <div class="subvalue">
                            {total_pdfs_existed} существовало, 
                            {total_pdfs_overwritten} перезаписано, 
                            {total_pdfs_error} ошибок
                        </div>
                        <div class="progress-bar">
                            <div class="progress" style="width: {pdf_success_rate:.1f}%"></div>
                        </div>
                    </div>
                    
                    <div class="card warning">
                        <h3>Дополнительные материалы</h3>
                        <div class="value">{total_supp}</div>
                        <div class="subvalue">
                            {total_supp_existed} существовало, 
                            {total_supp_overwritten} перезаписано, 
                            {total_supp_error} ошибок
                        </div>
                        <div class="progress-bar">
                            <div class="progress" style="width: {supp_success_rate:.1f}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Топ-10 мероприятий по количеству материалов</h2>
            <table>
                <thead>
                    <tr>
                        <th>Мероприятие</th>
                        <th>Версия API</th>
                        <th>Заметки</th>
                        <th>PDF</th>
                        <th>Доп. материалы</th>
                        <th>Время обработки</th>
                    </tr>
                </thead>
                <tbody>
    """

    for venue in top_venues:
        html_content += f"""
                    <tr>
                        <td>{venue.get("venue_id", "неизвестно")}</td>
                        <td>{venue.get("api_version", "неизвестно")}</td>
                        <td>{venue.get("notes_count", 0)}</td>
                        <td>{venue.get("pdf_count", 0)}</td>
                        <td>{venue.get("supplementary_count", 0)}</td>
                        <td>{venue.get("processing_time", 0):.1f}s</td>
                    </tr>
        """

    # Если есть ошибки, добавляем таблицу с ошибками
    if error_venues > 0:
        html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Мероприятия с ошибками</h2>
            <table>
                <thead>
                    <tr>
                        <th>Мероприятие</th>
                        <th>Версия API</th>
                        <th>Время обработки</th>
                    </tr>
                </thead>
                <tbody>
        """

        for venue in results["error"]:
            html_content += f"""
                    <tr>
                        <td>{venue.get("venue_id", "неизвестно")}</td>
                        <td>{venue.get("api_version", "неизвестно")}</td>
                        <td>{venue.get("processing_time", 0):.1f}s</td>
                    </tr>
            """

    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="timestamp">
            <p>Отчёт сгенерирован автоматически системой OpenReviewStore</p>
        </div>
        
        <script>
            // Анимация прогресс-баров
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(function() {
                    let progressBars = document.querySelectorAll('.progress');
                    progressBars.forEach(function(bar) {
                        let width = bar.style.width;
                        bar.style.width = '0%';
                        setTimeout(function() {
                            bar.style.width = width;
                        }, 100);
                    });
                }, 500);
            });
        </script>
    </body>
    </html>
    """

    # Создаем директорию для отчётов
    report_path = store_path / "reports"
    report_path.mkdir(exist_ok=True)

    # Сохраняем HTML-отчёт в файл
    report_file = report_path / report_filename
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Сохраняем сериализованные данные в JSON для возможного последующего анализа
    data_file = report_path / "statistics_data.json"
    with open(data_file, "w", encoding="utf-8") as f:
        # Создаем копию результатов с форматированием datetime
        serializable_results = {"success": [], "empty": [], "error": []}

        # Преобразование datetime в строки для JSON
        for status in ["success", "empty", "error"]:
            for item in results[status]:
                item_copy = item.copy()
                if "timestamp" in item_copy and isinstance(item_copy["timestamp"], datetime.datetime):
                    item_copy["timestamp"] = item_copy["timestamp"].isoformat()
                serializable_results[status].append(item_copy)

        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Отчёт сгенерирован: {report_file}")
    return report_file


def download_all(
    venue_ids: List[str],
    store_path: Path,
    max_workers: int = 5,
    overwrite_notes: bool = False,
    overwrite_pdfs: bool = False,
    overwrite_supplementary: bool = False,
    overwrite_venue: bool = False,
    debug: bool = False,
    generate_report: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Основная функция для скачивания данных с OpenReview.

    Args:
        venue_ids: Список идентификаторов мероприятий для обработки
        store_path: Путь для сохранения данных
        max_workers: Максимальное количество параллельных обработчиков
        overwrite_notes: Флаг перезаписи файлов заметок
        overwrite_pdfs: Флаг перезаписи PDF-файлов
        overwrite_supplementary: Флаг перезаписи дополнительных материалов
        overwrite_venue: Флаг перезаписи файлов информации о мероприятии
        debug: Флаг отладочного режима
        generate_report: Флаг генерации HTML-отчёта

    Returns:
        Dict[str, List[Dict[str, Any]]]: Словарь с результатами обработки мероприятий,
            где ключи: "success", "empty", "error", а значения - списки результатов
    """
    # Настройка логирования
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)

    # Создание директории для данных
    store_path.mkdir(parents=True, exist_ok=True)

    # Создаем файлы для отчетности
    report_path = store_path / "reports"
    report_path.mkdir(exist_ok=True)

    processed_file = report_path / "processed_venues.txt"
    empty_file = report_path / "empty_venues.txt"
    error_file = report_path / "error_venues.txt"

    clients = OpenReviewClients()

    if not venue_ids:
        logger.info("Получение списка всех мероприятий...")
        venue_ids = get_all_venues_name(clients.get_client("v2"))
        logger.info(f"Найдено {len(venue_ids)} мероприятий")
    else:
        logger.info(f"Обработка указанных мероприятий: {len(venue_ids)} шт.")
        if logger.isEnabledFor(logging.DEBUG):
            for v_id in venue_ids:
                logger.debug(f"Будет обработано: {v_id}")

    # Для хранения результатов
    results = {"success": [], "empty": [], "error": []}

    for i_venue, venue_id in enumerate(venue_ids):
        logger.info(f"[{i_venue + 1}/{len(venue_ids)}] Обработка мероприятия: {venue_id}")

        result = process_venue(
            venue_id=venue_id,
            clients=clients,
            store_path=store_path,
            max_workers=max_workers,
            overwrite_notes=overwrite_notes,
            overwrite_pdfs=overwrite_pdfs,
            overwrite_supplementary=overwrite_supplementary,
            overwrite_venue=overwrite_venue,
            report_file=processed_file,
        )

        # Сортировка результатов по статусу
        if result["status"] == "success":
            results["success"].append(result)
        elif result["status"] == "empty":
            results["empty"].append(result)
            with open(empty_file, "a") as f:
                f.write(f"{result['timestamp']}\t{venue_id}\t{result['api_version']}\n")
        elif result["status"] == "error":
            results["error"].append(result)
            with open(error_file, "a") as f:
                f.write(f"{result['timestamp']}\t{venue_id}\t{result['api_version'] or 'unknown'}\n")

    logger.info("Обработка завершена!")

    # Краткая статистика
    logger.info(f"Успешно обработано: {len(results['success'])} мероприятий")
    logger.info(f"Пустых мероприятий: {len(results['empty'])}")
    logger.info(f"Ошибок обработки: {len(results['error'])}")

    # Генерация HTML-отчёта
    if generate_report:
        report_file = generate_statistics_report(results, store_path)
        logger.info(f"Подробная статистика доступна в файле: {report_file}")

    return results


def generate_report_from_processed_data(store_path: Path) -> Path:
    """Генерирует отчёт на основе уже скаченных данных.

    Args:
        store_path: Путь к директории с данными

    Returns:
        Path: Путь к созданному отчёту
    """
    report_path = store_path / "reports"
    processed_file = report_path / "processed_venues.txt"
    empty_file = report_path / "empty_venues.txt"
    error_file = report_path / "error_venues.txt"

    # Проверяем наличие директории для отчетов
    if not report_path.exists():
        report_path.mkdir(parents=True, exist_ok=True)

    # Подготовка структуры данных для результатов
    results = {"success": [], "empty": [], "error": []}

    # Проверяем наличие хотя бы одного из файлов
    files_exist = any(
        [
            processed_file.exists() and processed_file.stat().st_size > 0,
            empty_file.exists() and empty_file.stat().st_size > 0,
            error_file.exists() and error_file.stat().st_size > 0,
        ]
    )

    if not files_exist:
        logger.warning("Файлы отчётов не найдены или пусты. Создаем пустой отчет.")

    # Читаем данные об успешно обработанных мероприятиях
    if processed_file.exists():
        with open(processed_file, "r") as f:
            lines = f.readlines()

            # Пропускаем заголовок
            if len(lines) > 0:
                header = lines[0].strip().split("\t")
                for line in lines[1:]:
                    parts = line.strip().split("\t")
                    if len(parts) < len(header):
                        continue

                    data = dict(zip(header, parts))

                    # Конвертация строковых значений в правильные типы
                    result = {
                        "timestamp": data.get("timestamp", ""),
                        "venue_id": data.get("venue_id", ""),
                        "status": data.get("status", ""),
                        "api_version": data.get("api_version", ""),
                        "notes_count": int(data.get("notes_count", "0")),
                        "pdf_count": int(data.get("pdf_count", "0")),
                        "supplementary_count": int(data.get("supplementary_count", "0")),
                        "notes_existed": int(data.get("notes_existed", "0")),
                        "notes_overwritten": int(data.get("notes_overwritten", "0")),
                        "notes_error": int(data.get("notes_error", "0")),
                        "pdfs_existed": int(data.get("pdfs_existed", "0")),
                        "pdfs_overwritten": int(data.get("pdfs_overwritten", "0")),
                        "pdfs_error": int(data.get("pdfs_error", "0")),
                        "supps_existed": int(data.get("supps_existed", "0")),
                        "supps_overwritten": int(data.get("supps_overwritten", "0")),
                        "supps_error": int(data.get("supps_error", "0")),
                    }

                    # Время обработки (убираем 's' в конце)
                    processing_time = data.get("processing_time", "0s")
                    if processing_time.endswith("s"):
                        processing_time = processing_time[:-1]
                    result["processing_time"] = float(processing_time)

                    # Добавляем в соответствующий список по статусу
                    status = data.get("status", "")
                    if status == "success":
                        results["success"].append(result)
                    elif status == "empty":
                        results["empty"].append(result)

    # Читаем данные о пустых мероприятиях, если нет данных в processed_file
    if empty_file.exists() and not any(r.get("status") == "empty" for r in results["empty"]):
        with open(empty_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    timestamp, venue_id = parts[0], parts[1]
                    api_version = parts[2] if len(parts) > 2 else "unknown"

                    results["empty"].append(
                        {
                            "timestamp": timestamp,
                            "venue_id": venue_id,
                            "status": "empty",
                            "api_version": api_version,
                            "processing_time": 0,
                        }
                    )

    # Читаем данные о мероприятиях с ошибками
    if error_file.exists():
        with open(error_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    timestamp, venue_id = parts[0], parts[1]
                    api_version = parts[2] if len(parts) > 2 else "unknown"

                    results["error"].append(
                        {
                            "timestamp": timestamp,
                            "venue_id": venue_id,
                            "status": "error",
                            "api_version": api_version,
                            "processing_time": 0,
                        }
                    )

    logger.info(f"Найдено успешных мероприятий: {len(results['success'])}")
    logger.info(f"Найдено пустых мероприятий: {len(results['empty'])}")
    logger.info(f"Найдено мероприятий с ошибками: {len(results['error'])}")

    # Генерируем и возвращаем отчёт
    return generate_statistics_report(results, store_path)


def cli():
    """Функция командной строки для запуска скрипта."""
    args = parse_args()

    # Настраиваем логирование для вывода в консоль
    log_level = logging.DEBUG if args.debug else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    # Настраиваем логгер модуля
    logger.setLevel(log_level)

    # Создание Path объекта из строки пути
    store_path = Path(args.store_path)

    # Проверяем флаг генерации только отчёта
    if args.report_only:
        logger.info("Режим генерации отчёта без скачивания данных")
        report_file = generate_report_from_processed_data(store_path)
        if report_file:
            logger.info(f"Отчёт сгенерирован: {report_file}")
        else:
            logger.error("Не удалось сгенерировать отчёт. Проверьте наличие данных в указанной директории.")
        return

    # Определение флагов перезаписи
    overwrite_notes = args.overwrite_notes or args.overwrite_all
    overwrite_pdfs = args.overwrite_pdfs or args.overwrite_all
    overwrite_supplementary = args.overwrite_supplementary or args.overwrite_all
    overwrite_venue = args.overwrite_venue or args.overwrite_all

    # Вызов основной функции
    download_all(
        venue_ids=args.venue_id,
        store_path=store_path,
        max_workers=args.max_workers,
        overwrite_notes=overwrite_notes,
        overwrite_pdfs=overwrite_pdfs,
        overwrite_supplementary=overwrite_supplementary,
        overwrite_venue=overwrite_venue,
        debug=args.debug,
        generate_report=not args.no_report,
    )


def main():
    """Устаревшая функция для обратной совместимости."""
    cli()


if __name__ == "__main__":
    cli()

"""
Пример использования функции download_all из других модулей:

```python
from pathlib import Path
from openreviewstore.parsing import download_all

# Скачать определенные мероприятия
results = download_all(
    venue_ids=["ICLR.cc/2023/Conference", "NeurIPS.cc/2022/Conference"],
    store_path=Path("./data/my_store"),
    max_workers=8,
    overwrite_pdfs=True
)

# Вывести статистику
print(f"Успешно обработано: {len(results['success'])} мероприятий")
print(f"Ошибок: {len(results['error'])} мероприятий")

# Работа с результатами
for venue_result in results['success']:
    print(f"Мероприятие: {venue_result['venue_id']}")
    print(f"  Найдено заметок: {venue_result['notes_count']}")
    print(f"  Скачано PDF: {venue_result['pdf_count']}")
    print(f"  Скачано дополнительных материалов: {venue_result['supplementary_count']}")
```

Пример использования отчётов через командную строку:

```bash
# Скачать данные и сгенерировать отчёт
python -m openreviewstore.parsing.download_all --venue-id ICLR.cc/2023/Conference --store-path ./data/my_store

# Только сгенерировать отчёт на основе уже скаченных данных
python -m openreviewstore.parsing.download_all --report-only --store-path ./data/my_store

# Скачать данные без генерации отчёта
python -m openreviewstore.parsing.download_all --venue-id ICLR.cc/2023/Conference --no-report
```
"""
