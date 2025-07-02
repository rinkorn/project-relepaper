from typing import List

from loguru import logger
from pyalex import Works, config

from relepaper.domains.openalex.entities.work import OpenAlexWork
from relepaper.domains.openalex.external.interfaces import IAdapter


class PyAlexWorksSearchAdapter(IAdapter):
    """Adapter for PyAlex library."""

    def __init__(self, email: str = "mail@example.com", max_retries: int = 3, retry_backoff_factor: float = 0.5):
        self.configure(email, max_retries, retry_backoff_factor)

    def configure(self, email: str, max_retries: int = 3, retry_backoff_factor: float = 0.5) -> None:
        """Configure PyAlex."""
        config.email = email
        config.max_retries = max_retries
        config.retry_backoff_factor = retry_backoff_factor

    def search_works(self, query: str, per_page: int = 5) -> List[OpenAlexWork]:
        """Search for articles using PyAlex.

        Args:
            query: user query
            per_page: number of works to return
        Returns:
            list of OpenAlexWork objects
        """
        logger.trace(f"{self.__class__.__name__}: search_works: start")
        logger.debug(f"{self.__class__.__name__}: search_works: query: {query}")
        W = Works().search_filter(title_and_abstract=query)
        W = W.filter(has_oa_accepted_or_published_version=True)
        W = W.sort(cited_by_count="desc")
        works = W.get(per_page=per_page)
        open_alex_works = [OpenAlexWork.from_dict(work) for work in works]
        logger.info(f"{self.__class__.__name__}: search_works: found works: {len(open_alex_works)}")
        logger.trace(f"{self.__class__.__name__}: search_works: end")
        return open_alex_works
