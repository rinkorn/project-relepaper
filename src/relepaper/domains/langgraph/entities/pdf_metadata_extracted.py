from dataclasses import dataclass
from pprint import pformat
from typing import List


@dataclass
class PDFMetadataExtracted:
    title: str
    abstract: str
    year: int
    authors: List[str]
    keywords: List[str]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" + pformat(self.__dict__) + ")"

    def __repr__(self) -> str:
        return self.__str__()


# %%
if __name__ == "__main__":
    metadata = PDFMetadataExtracted(
        title="Hello, world!",
        abstract="This is a test abstract with a lot of text, and it is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very long text. It is a very",
        authors=["John Doe"],
        year=2021,
        keywords=["hello", "world"],
    )
    print(metadata)
