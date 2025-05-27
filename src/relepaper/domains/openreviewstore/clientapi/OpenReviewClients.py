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
        self.v1_baseurl = v1_baseurl
        self.v2_baseurl = v2_baseurl
        self.username = username
        self.password = password
        self._init_clients()

    def _init_clients(self):
        # Инициализация клиентов с авторизацией или без
        if self.username and self.password:
            self._clients["v1"] = openreview.Client(
                baseurl=self.v1_baseurl,
                username=self.username,
                password=self.password,
            )
            self._clients["v2"] = openreview.api.OpenReviewClient(
                baseurl=self.v2_baseurl,
                username=self.username,
                password=self.password,
            )
            logger.info("Инициализированы клиенты API v1 и v2 с авторизацией")
        else:
            self._clients["v1"] = openreview.Client(baseurl=self.v1_baseurl)
            self._clients["v2"] = openreview.api.OpenReviewClient(baseurl=self.v2_baseurl)
            logger.info("Инициализированы клиенты API v1 и v2 без авторизации")

    def get_client(self, api_version: str):
        """
        Получает клиент API по указанной версии.

        Args:
            api_version: Версия API ("v1" или "v2")

        Returns:
            Клиент API соответствующей версии

        Raises:
            ValueError: Если указана неверная версия API
        """
        if api_version not in self._clients:
            raise ValueError(f"Неверная версия API: {api_version}. Доступные версии: {list(self._clients.keys())}")
        return self._clients[api_version]


if __name__ == "__main__":
    clients = OpenReviewClients()
    client = clients.get_client("v2")
