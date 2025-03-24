# %%
import logging

import openreview

from .OpenReviewClients import OpenReviewClients

logger = logging.getLogger(__name__)


def identify_client_for_venue(clients: OpenReviewClients, venue_id: str) -> str:
    """
    Определяет, какая версия API должна использоваться для указанной площадки.

    Проверяет наличие поля domain в группе площадки. Если поле отсутствует,
    то площадка доступна через API v1, иначе - через API v2.

    Args:
        clients: Экземпляр класса OpenReviewClients
        venue_id: Идентификатор площадки

    Returns:
        Строка "v1" или "v2", указывающая нужную версию API

    Raises:
        openreview.OpenReviewException: При ошибке доступа к группе площадки
    """
    logger.info(f"Определение версии API для площадки: {venue_id}")

    # По умолчанию используем API v2
    client_version = "v2"

    try:
        # Пробуем получить группу площадки через API v2
        venue_group = clients.get_client("v2").get_group(venue_id)

        # Проверяем наличие поля domain
        if venue_group.domain is None:
            logger.info(f"Площадка {venue_id} доступна через API v1 (отсутствует поле domain)")
            client_version = "v1"
        else:
            logger.info(f"Площадка {venue_id} доступна через API v2 (имеет поле domain)")
    except openreview.OpenReviewException as e:
        logger.error(f"Ошибка при определении версии API для {venue_id}: {str(e)}")
        # Если не удалось получить группу через API v2, пробуем через API v1
        try:
            clients.get_client("v1").get_group(venue_id)
            logger.info(f"Площадка {venue_id} доступна через API v1")
            client_version = "v1"
        except openreview.OpenReviewException as e2:
            logger.error(f"Ошибка при проверке API v1 для {venue_id}: {str(e2)}")
            raise ValueError(f"Не удалось определить версию API для площадки {venue_id}")
    return client_version
