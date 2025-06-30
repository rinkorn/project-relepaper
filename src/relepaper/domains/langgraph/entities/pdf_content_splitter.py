from pprint import pprint
from typing import List

from loguru import logger

from relepaper.domains.langgraph.entities.pdf_content_chunk import PDFContentChunk
from relepaper.domains.pdf_exploring.entities.pdf_document import PDFDocument


# %%
class PDFContentSplitter:
    def __init__(
        self,
        pdf_document: PDFDocument,
        max_chunk_length: int = 10000,
        max_chunks_count: int | None = None,
        intersection_length: int = 0,
    ):
        if intersection_length >= max_chunk_length:
            raise ValueError("Intersection length must be less than max part length")

        self._pdf_document = pdf_document
        self._max_chunk_length = max_chunk_length
        self._max_chunks_count = max_chunks_count
        self._intersection_length = intersection_length
        self._chunks = []
        self._split_pdf_document()

    def _split_pdf_document(self):
        logger.trace("PDFContentSplitter: _split_pdf_document: start")
        text = self._pdf_document.text.text
        chunks = []
        for i in range(0, len(text), self._max_chunk_length - self._intersection_length):
            chunks.append(
                PDFContentChunk(
                    text[i : i + self._max_chunk_length], i // (self._max_chunk_length - self._intersection_length)
                )
            )
            if self._max_chunks_count is not None and len(chunks) >= self._max_chunks_count:
                break
        self._chunks = chunks
        logger.trace("PDFContentSplitter: _split_pdf_document: end")

    def get_chunks(self) -> List[PDFContentChunk]:
        return self._chunks

    def __getitem__(self, index: int) -> PDFContentChunk:
        return self._chunks[index]

    def __len__(self) -> int:
        return len(self._chunks)


if __name__ == "__main__":
    from relepaper.domains.pdf_exploring.entities.pdf_document import PDFDocument, PDFMetadata, PDFText

    pdf_document = PDFDocument(
        text=PDFText(text="Hello, world! This is a test of the PDFText class."),
        metadata=PDFMetadata(
            title="Test",
            authors=["John Doe"],
            keywords=["test", "test2"],
        ),
    )
    splitter = PDFContentSplitter(
        pdf_document,
        max_chunk_length=10,
        max_chunks_count=None,
        intersection_length=2,
    )
    print(len(splitter))
    pprint(splitter.get_chunks())
    print(splitter[0])
    print(len(splitter[0]))
    print(splitter[1])
    print(len(splitter[1]))
