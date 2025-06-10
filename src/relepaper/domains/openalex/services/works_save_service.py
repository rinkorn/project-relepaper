from typing import List

from relepaper.domains.openalex.entities.work import OpenAlexWork
from relepaper.domains.openalex.external.interfaces import IRepository
from relepaper.domains.openalex.services.interfaces import IService


class OpenAlexWorksSaveService(IService):
    def __init__(self, repository: IRepository, **kwargs):
        self._repository = repository
        self._kwargs = kwargs

    def save_work(self, work: OpenAlexWork):
        self._repository.save(work)

    def save_works(self, works: List[OpenAlexWork]):
        self._repository.save_all(works)
