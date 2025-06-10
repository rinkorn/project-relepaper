from typing import List

from relepaper.domains.openalex.entities.work import OpenAlexWork
from relepaper.domains.openalex.external.adapters.works_search.factory import OpenAlexWorksSearchAdapterFactory
from relepaper.domains.openalex.external.interfaces import IAdapter
from relepaper.domains.openalex.services.interfaces import IService


class OpenAlexWorksSearchService(IService):
    def __init__(self, adapter: IAdapter = None):
        if adapter is None:
            # Use factory to create default adapter
            adapter = OpenAlexWorksSearchAdapterFactory.create(
                strategy="pyalex",
                config={
                    "email": "mail@example.com",
                    "max_retries": 3,
                    "retry_backoff_factor": 0.5,
                },
            )
        self._adapter = adapter

    def search_works(self, query: str, per_page: int = 5) -> List[OpenAlexWork]:
        """Search for articles on the OpenAlex hub.

        Args:
            query: user query
            per_page: number of works to return
        Returns:
            list of OpenAlexWork objects
        """
        return self._adapter.search_works(query, per_page)
