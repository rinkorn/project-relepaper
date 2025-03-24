# %%
import logging
from typing import Optional

import openreview

logger = logging.getLogger(__name__)


class OpenReviewClients:
    """
    Класс для управления клиентами OpenReview API разных версий.
    """

    def __init__(
        self,
        v1_baseurl: str = "https://api.openreview.net",
        v2_baseurl: str = "https://api2.openreview.net",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Инициализирует клиенты OpenReview API v1 и v2.

        Args:
            v1_baseurl: URL для API v1
            v2_baseurl: URL для API v2
            username: Имя пользователя для авторизации (опционально)
            password: Пароль для авторизации (опционально)
        """
        self._clients = {}

        # Инициализация клиентов с авторизацией или без
        if username and password:
            self._clients["v1"] = openreview.Client(
                baseurl=v1_baseurl,
                username=username,
                password=password,
            )
            self._clients["v2"] = openreview.api.OpenReviewClient(
                baseurl=v2_baseurl,
                username=username,
                password=password,
            )
            logger.info("Инициализированы клиенты API v1 и v2 с авторизацией")
        else:
            self._clients["v1"] = openreview.Client(baseurl=v1_baseurl)
            self._clients["v2"] = openreview.api.OpenReviewClient(baseurl=v2_baseurl)
            logger.info("Инициализированы клиенты API v1 и v2 без авторизации")

    def get_client(self, version: str):
        """
        Получает клиент API по указанной версии.

        Args:
            version: Версия API ("v1" или "v2")

        Returns:
            Клиент API соответствующей версии

        Raises:
            ValueError: Если указана неверная версия API
        """
        if version not in self._clients:
            raise ValueError(f"Неверная версия API: {version}. Доступные версии: {list(self._clients.keys())}")
        return self._clients[version]
