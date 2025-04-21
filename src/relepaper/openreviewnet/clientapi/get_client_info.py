# %%
import logging
from typing import Dict

import openreview

from relepaper.openreviewnet.clientapi.get_current_user_info import get_current_user_info
from relepaper.openreviewnet.clientapi.get_profile_by_id import get_profile_by_id

logger = logging.getLogger(__name__)


def get_client_info(client) -> Dict[str, str]:
    """
    Получает информацию о клиенте OpenReview.

    Args:
        client: Экземпляр клиента OpenReview

    Returns:
        Dict[str, str]: Словарь с информацией о клиенте
    """
    return {
        "baseurl": client.baseurl,
        "авторизован": "Да" if client.token else "Нет",
    }


if __name__ == "__main__":
    from relepaper.openreviewnet.clientapi.OpenReviewClients import OpenReviewClients

    clients = OpenReviewClients()
    client = clients.get_client("v2")

    # Информация о клиенте
    client_info = get_client_info(client)
    print("Информация о клиенте:")
    for key, value in client_info.items():
        print(f"{key}: {value}")

    # Информация о текущем пользователе
    user_info = get_current_user_info(client)
    print("\nИнформация о пользователе:")
    for key, value in user_info.items():
        if key != "профиль":  # Пропускаем полный профиль для краткости вывода
            print(f"{key}: {value}")

    # Если пользователь авторизован, можно попробовать получить его профиль по ID
    if user_info.get("авторизован"):
        try:
            profile_info = get_profile_by_id(client, user_info["id"])
            if profile_info:
                print("\nПрофиль пользователя (получен по ID):")
                for key, value in profile_info.items():
                    if key != "профиль":  # Пропускаем полный профиль для краткости вывода
                        print(f"{key}: {value}")
        except openreview.OpenReviewException as e:
            print(f"Ошибка при получении профиля: {str(e)}")
