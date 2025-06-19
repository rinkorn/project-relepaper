from dataclasses import dataclass
from typing import List


@dataclass
class ExtractedPDFMetadata:
    title: str
    abstract: str
    year: int
    authors: List[str]
    keywords: List[str]
    num_pages: int

    def __str__(self) -> str:
        return f"ExtractedPDFMetadata(\n  title={self.title},\n  authors={self.authors},\n  year={self.year},\n  keywords={self.keywords},\n  num_pages={self.num_pages}\n)"

    def __repr__(self) -> str:
        return self.__str__()


# %%
if __name__ == "__main__":
    metadata = ExtractedPDFMetadata(
        title="Hello, world!",
        abstract="This is a test abstract",
        authors=["John Doe"],
        year=2021,
        keywords=["hello", "world"],
        num_pages=100,
    )
    print(metadata)
