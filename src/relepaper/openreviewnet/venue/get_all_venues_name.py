# %%
import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def get_all_venues_name(client: Any) -> List[str]:
    """Получает список всех мероприятий.

    Args:
        client: Объект клиента OpenReview

    Returns:
        List[str]: Список всех мероприятий
    """
    try:
        venues = client.get_group(id="venues").members
        return venues
    except Exception as e:
        logger.error(f"Ошибка при получении списка мероприятий: {str(e)}")
        return []


# %%
if __name__ == "__main__":
    from relepaper.openreviewnet.clientapi.OpenReviewClients import OpenReviewClients

    clients = OpenReviewClients()
    venues = get_all_venues_name(clients.get_client("v2"))

    print("\n".join(venues[:10]))
