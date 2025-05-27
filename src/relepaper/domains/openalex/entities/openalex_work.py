from dataclasses import dataclass
from typing import List


@dataclass
class OpenAlexWork:
    url_id: str
    doi: str
    title: str
    authors: List[str]
    year: int
    keywords: List[str]
    abstract: str
    primary_location: dict
    additional_kwargs: dict

    @property
    def pdf_url(self) -> str | None:
        return self.primary_location.get("pdf_url")

    @property
    def id(self) -> str:
        if self.url_id.startswith("https://openalex.org/"):
            return self.url_id.split("/")[-1]
        return self.url_id

    def from_external(self, external_work: dict) -> "OpenAlexWork":
        url_id = external_work.pop("id", None)
        title = external_work.pop("title", None)
        authors = external_work.pop("authors", None)
        year = external_work.pop("year", None)
        doi = external_work.pop("doi", None)
        keywords = external_work.pop("keywords", None)
        abstract = external_work.pop("abstract", None)
        primary_location = external_work.pop("primary_location", {})

        return OpenAlexWork(
            url_id=url_id,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            keywords=keywords,
            abstract=abstract,
            primary_location=primary_location,
            additional_kwargs=external_work,
        )
