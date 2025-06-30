from dataclasses import dataclass
from typing import List


@dataclass
class PDFText:
    text: str

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, index: int) -> str:
        return self.text[index]

    def __slice__(self, start: int, end: int) -> str:
        return self.text[start:end]


@dataclass
class PDFImage:
    image: bytes


@dataclass
class PDFMetadata:
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    authors: List[str] | None = None
    keywords: List[str] | None = None
    num_pages: int | None = None

    def __str__(self) -> str:
        return (
            f"PDFMetadata(\n"
            f"  title={self.title},\n"
            f"  abstract={self.abstract},\n"
            f"  year={self.year},\n"
            f"  authors={self.authors},\n"
            f"  keywords={self.keywords},\n"
            f"  num_pages={self.num_pages}\n)"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class PDFDocument:
    metadata: PDFMetadata
    text: PDFText
    images: List[PDFImage] | None = None

    def __str__(self) -> str:
        text = self.text[:20] + "..." + self.text[-20:] if len(self.text) > 40 else self.text
        num_images = len(self.images) if self.images else None
        return f"PDFDocument(\n  metadata={self.metadata},\n  text={text},\n  num_images={num_images}\n)"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.text)


# %%
if __name__ == "__main__":
    metadata = PDFMetadata(
        title="Hello, world!",
        authors=["John Doe"],
        keywords=["test", "test", "test"],
        year=2021,
        abstract="Test abstract",
    )
    text = PDFText(text="Hello, world! This is a test of the PDFText class.")
    pdf_document = PDFDocument(
        metadata=metadata,
        text=text,
    )
    print(pdf_document)
    print(pdf_document.text[0])
    print(pdf_document.text[0:5])
    print(pdf_document.text[-5:])
    print(len(pdf_document.text))

# %%
