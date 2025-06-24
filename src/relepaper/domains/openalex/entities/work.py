# %%
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class OpenAlexWork:
    id: str
    doi: Optional[str] = None
    title: Optional[str] = None
    keywords: Optional[List[str]] = None
    abstract: Optional[str] = None
    primary_location: Optional[dict] = None
    additional_kwargs: Optional[dict] = None

    # Новые поля для связи с запросами
    source_query: Optional[str] = None
    source_query_index: Optional[int] = None
    found_at: Optional[datetime] = None

    def __post_init__(self):
        """Validation after initialization."""
        if not self.id:
            raise ValueError("id is required")

    @property
    def pdf_url(self) -> str | None:
        if self.primary_location is None:
            return None
        return self.primary_location.get("pdf_url")

    @property
    def clean_id(self) -> str:
        if self.id and self.id.startswith("https://openalex.org/"):
            return self.id.split("/")[-1]
        return self.id or ""

    @classmethod
    def from_dict(cls, work_data: dict) -> "OpenAlexWork":
        # make a copy to not change the original dictionary
        work_data = work_data.copy()

        id = work_data.pop("id", "")
        title = work_data.pop("title", None)
        doi = work_data.pop("doi", None)
        keywords = work_data.pop("keywords", None)
        abstract = work_data.pop("abstract", None)
        primary_location = work_data.pop("primary_location", None)

        # Извлекаем новые поля, если они есть
        source_query = work_data.pop("source_query", None)
        source_query_index = work_data.pop("source_query_index", None)
        found_at = work_data.pop("found_at", None)

        # Обработка found_at если это строка
        if isinstance(found_at, str):
            try:
                found_at = datetime.fromisoformat(found_at)
            except ValueError:
                found_at = None

        return cls(
            id=id,
            title=title,
            doi=doi,
            keywords=keywords,
            abstract=abstract,
            primary_location=primary_location,
            source_query=source_query,
            source_query_index=source_query_index,
            found_at=found_at,
            additional_kwargs=work_data,  # other fields
        )

    @classmethod
    def model_validate(cls, data: dict) -> "OpenAlexWork":
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doi": self.doi,
            "title": self.title,
            "keywords": self.keywords,
            "abstract": self.abstract,
            "primary_location": self.primary_location,
            "source_query": self.source_query,
            "source_query_index": self.source_query_index,
            "found_at": self.found_at.isoformat() if self.found_at else None,
            "additional_kwargs": self.additional_kwargs,
        }

    def copy(self, update: dict) -> "OpenAlexWork":
        work = deepcopy(self)
        for key, value in update.items():
            setattr(work, key, value)
        return work


# %%
if __name__ == "__main__":
    from pprint import pprint

    reformulated_queries = [
        "fluorescent mobility experiments with significant equipment",
        "research on fluorescent mobility using advanced equipment",
        "studies on fluorescent mobility with high-impact tools",
        "investigations into fluorescent mobility through advanced instrumentation",
        "experiments on fluorescent mobility utilizing state-of-the-art equipment",
    ]

    from pyalex import Works, config

    config.email = "mail@example.com"
    config.max_retries = 3
    config.retry_backoff_factor = 0.5

    works: List[OpenAlexWork] = []
    for query in reformulated_queries[:1]:
        W = Works().search_filter(title_and_abstract=query)
        W = W.filter(has_oa_accepted_or_published_version=True)
        W = W.sort(cited_by_count="desc")
        works_ = W.get(per_page=5)
        works.extend([OpenAlexWork.from_dict(work) for work in works_])

    # safe getting URL
    urls = [work.pdf_url for work in works if work.pdf_url is not None]
    pprint(f"URLs found: {urls}")
    pprint(f"Total URLs: {len(urls)}")
    pprint(f"Unique URLs: {len(set(urls))}")
    pprint(f"Total works: {len(works)}")
    pprint(f"Unique work IDs: {len(set([work.id for work in works]))}")


# %%
