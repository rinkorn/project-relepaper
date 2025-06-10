import abc
from abc import abstractmethod
from typing import List

from relepaper.domains.openalex.entities.work import OpenAlexWork


class IAdapter(abc.ABC):
    """Base interface for all adapters."""

    pass


class IAdapterFactory(abc.ABC):
    """Base interface for all factories."""

    @staticmethod
    @abc.abstractmethod
    def create() -> IAdapter:
        pass


class IRepository(abc.ABC):
    """Base interface for all repositories."""

    @abstractmethod
    def get_by_id(self, id: str) -> OpenAlexWork:
        pass

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[OpenAlexWork]:
        pass

    @abstractmethod
    def save(self, work: OpenAlexWork) -> None:
        pass

    @abstractmethod
    def save_all(self, works: List[OpenAlexWork]) -> None:
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        pass


class IRepositoryFactory(abc.ABC):
    """Base interface for all repositories factories."""

    @staticmethod
    @abc.abstractmethod
    def create() -> IRepository:
        pass
