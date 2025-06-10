from typing import Dict, List

from relepaper.domains.openalex.entities.work import OpenAlexWork
from relepaper.domains.openalex.external.interfaces import IRepository


class InMemoryWorksRepository(IRepository):
    def __init__(self):
        self._works: Dict[str, OpenAlexWork] = {}

    def get_by_id(self, id: str) -> OpenAlexWork:
        work = self._works.get(id)
        if not work:
            raise ValueError(f"Work with id {id} not found")
        return work

    def get_by_ids(self, ids: List[str]) -> List[OpenAlexWork]:
        works = [self._works.get(id) for id in ids if id in self._works]
        if not works:
            raise ValueError(f"Works with ids {ids} not found")
        return works

    def save(self, work: OpenAlexWork) -> None:
        self._works[work.id] = work

    def save_all(self, works: List[OpenAlexWork]) -> None:
        for work in works:
            self.save(work)

    def delete(self, id: str) -> None:
        self._works.pop(id, None)


if __name__ == "__main__":
    work = OpenAlexWork(id="1234567890", title="Test Work")
    works = [work.copy(update={"id": f"1234567890-{i}"}) for i in range(10)]
    repository = InMemoryWorksRepository()
    repository.save(work)
    repository.save_all(works)
    work = repository.get_by_id("1234567890")
    print(work)
    works = repository.get_by_ids([work.id])
    print(works)
    repository.delete(work.id)
    try:
        work = repository.get_by_id(work.id)
        print(work)
    except ValueError as e:
        print(e)
