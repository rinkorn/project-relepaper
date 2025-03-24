# %%
import logging
from typing import List, Tuple

from .OpenReviewClients import OpenReviewClients
from .identify_client_for_venue import identify_client_for_venue

logger = logging.getLogger(__name__)


def get_notes_for_venue(clients: OpenReviewClients, venue_id: str) -> Tuple[str, List]:
    """
    Получает заметки для указанной площадки, используя подходящую версию API.

    Args:
        clients: Экземпляр класса OpenReviewClients
        venue_id: Идентификатор площадки

    Returns:
        Кортеж (версия_api, список_заметок)
    """
    # Определяем подходящую версию API
    api_version = identify_client_for_venue(clients, venue_id)

    # Получаем клиент соответствующей версии
    client = clients.get_client(api_version)

    # Получаем заметки
    logger.info(f"Получение заметок для площадки {venue_id} через API {api_version}")
    try:
        notes = client.get_all_notes(content={"venueid": venue_id})
        logger.info(f"Найдено {len(notes)} заметок для площадки {venue_id}")
        return api_version, notes
    except Exception as e:
        logger.error(f"Ошибка при получении заметок для площадки {venue_id}: {str(e)}")
        return api_version, []
