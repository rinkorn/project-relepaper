# %%
import logging
from typing import Any, Dict, Optional

import openreview

logger = logging.getLogger(__name__)


def get_profile_by_id(client, profile_id: str) -> Optional[Dict[str, Any]]:
    """
    Получает информацию о профиле пользователя по ID.

    Args:
        client: Экземпляр клиента OpenReview
        profile_id (str): ID профиля пользователя

    Returns:
        Optional[Dict[str, Any]]: Словарь с информацией о пользователе или None, если профиль не найден

    Raises:
        openreview.OpenReviewException: Если возникла ошибка при получении профиля
    """
    try:
        profile = client.get_profile(profile_id)
        if not profile:
            return None

        return {
            "id": profile.id,
            "имя": profile.content.get("names", [{}])[0].get("fullname", "Не указано"),
            "email": profile.content.get("emails", ["Не указано"])[0],
            "институт": profile.content.get("history", [{}])[0].get("institution", {}).get("value", "Не указан"),
            "профиль": profile,
        }
    except openreview.OpenReviewException as e:
        raise openreview.OpenReviewException(f"Ошибка при получении профиля {profile_id}: {str(e)}")
