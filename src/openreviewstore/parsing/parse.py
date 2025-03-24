# %%
import argparse
import logging
import sys

from site_parser.storing.get_notes_for_venue import get_notes_for_venue
from site_parser.storing.OpenReviewClients import OpenReviewClients
from site_parser.storing.simplify_note_content import simplify_note_content

logger = logging.getLogger(__name__)


def parse_args():
    """
    Разбирает аргументы командной строки.

    Returns:
        Объект с разобранными аргументами
    """
    parser = argparse.ArgumentParser(description="Определяет версию API OpenReview для площадки и получает заметки.")
    parser.add_argument("venues", nargs="+", help="Идентификаторы площадок для проверки")
    parser.add_argument("--username", help="Имя пользователя для авторизации в OpenReview")
    parser.add_argument("--password", help="Пароль для авторизации в OpenReview")
    parser.add_argument(
        "--v1-baseurl",
        default="https://api.openreview.net",
        help="URL для API v1 (по умолчанию: https://api.openreview.net)",
    )
    parser.add_argument(
        "--v2-baseurl",
        default="https://api2.openreview.net",
        help="URL для API v2 (по умолчанию: https://api2.openreview.net)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    return parser.parse_args()


def main():
    """
    Основная функция для запуска скрипта из командной строки.
    """
    args = parse_args()

    # Настройка уровня логирования
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # logging.basicConfig(
    #     level=log_level,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     stream=sys.stdout,
    # )
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    

    # Инициализация клиентов
    clients = OpenReviewClients(
        v1_baseurl=args.v1_baseurl,
        v2_baseurl=args.v2_baseurl,
        username=args.username,
        password=args.password,
    )

    # Обработка каждой площадки
    for venue_id in args.venues:
        try:
            # Определяем версию API и получаем заметки
            api_version, notes = get_notes_for_venue(clients, venue_id)

            print(f"Площадка: {venue_id} - API: {api_version}")
            print(f"Найдено заметок: {len(notes)}")

            # Выводим информацию о первых 5 заметках
            for i, note in enumerate(notes[:5]):
                note_data = simplify_note_content(note.to_json())
                title = note_data.get("content", {}).get("title", "Нет заголовка")
                note_id = note_data.get("id", "Нет ID")
                print(f"  {i + 1}. {note_id} - {title}")

            if len(notes) > 5:
                print(f"  ... и еще {len(notes) - 5} заметок")

            print("")
        except Exception as e:
            logger.error(f"Ошибка при обработке площадки {venue_id}: {str(e)}")
            print(f"Ошибка для площадки {venue_id}: {str(e)}")


if __name__ == "__main__":
    main()
