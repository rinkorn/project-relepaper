import openreview
from typing import Dict, Any, Optional


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
                "профиль": profile
            }
        else:
            return {"авторизован": False, "сообщение": "Профиль не найден"}
    except openreview.OpenReviewException as e:
        return {"авторизован": False, "сообщение": f"Не авторизован: {str(e)}"}


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
            "профиль": profile
        }
    except openreview.OpenReviewException as e:
        raise openreview.OpenReviewException(f"Ошибка при получении профиля {profile_id}: {str(e)}")


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
        "авторизован": "Да" if client.token else "Нет"
    }


if __name__ == "__main__":
    # Пример использования
    # client = openreview.Client(baseurl="https://api2.openreview.net")
    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
    
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