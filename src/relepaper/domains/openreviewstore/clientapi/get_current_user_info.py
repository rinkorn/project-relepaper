# %%
import logging
from typing import Any, Dict

import openreview

logger = logging.getLogger(__name__)


def get_current_user_info(client) -> Dict[str, Any]:
    """
    Функция проверяет авторизован ли пользователь в OpenReview
    и возвращает информацию о текущем пользователе.

    Args:
        client: Экземпляр клиента OpenReview

    Returns:
        Dict[str, Any]: Словарь с информацией о пользователе или статусе неавторизованности

    Raises:
        openreview.OpenReviewException: Если возникла ошибка при получении профиля
    """
    try:
        profile = client.get_profile()
        if profile:
            return {
                "авторизован": True,
                "id": profile.id,
                "имя": profile.content.get("names", [{}])[0].get("fullname", "Не указано"),
                "email": profile.content.get("emails", ["Не указано"])[0],
                "профиль": profile,
            }
        else:
            return {
                "авторизован": False,
                "сообщение": "Профиль не найден",
            }
    except openreview.OpenReviewException as e:
        return {
            "авторизован": False,
            "сообщение": f"Не авторизован: {str(e)}",
        }
