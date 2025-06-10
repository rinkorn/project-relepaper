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
    author: str | None = None
    subject: str | None = None
    keywords: List[str] | None = None
    creator: str | None = None
    producer: str | None = None
    creationDate: str | None = None
    modDate: str | None = None
    trapped: str | None = None
    encryption: str | None = None
    num_pages: int | None = None


@dataclass
class PDFDocument:
    metadata: PDFMetadata
    text: PDFText
    images: list[PDFImage] | None = None

    def __str__(self) -> str:
        text = self.text[:20] + "..." + self.text[-20:] if len(self.text) > 40 else self.text
        num_images = len(self.images) if self.images else 0
        return f"PDFDocument(metadata={self.metadata},text={text},num_images={num_images})"

    def __repr__(self) -> str:
        return self.__str__()


# %%
if __name__ == "__main__":
    metadata = PDFMetadata(
        title="Hello, world!",
        author="John Doe",
        subject="Test",
        keywords="test, test, test",
        creator="John Doe",
        producer="John Doe",
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
