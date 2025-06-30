from dataclasses import dataclass
from pprint import pformat


# %%
@dataclass
class PDFContentChunk:
    text: str
    chunk_number: int

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            + "("
            + pformat(
                {
                    "text": self.text[:100] + "... . . . ..." + self.text[-100:] if len(self.text) > 200 else self.text,
                    "length": len(self.text),
                    "chunk_number": self.chunk_number,
                }
            )
            + ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


# %%
if __name__ == "__main__":
    chunk = PDFContentChunk(
        text="Hello, world! This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDF",
        chunk_number=0,
    )
    chunk2 = PDFContentChunk(
        text="Hello, world! This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDFContentChunk class. This is a test of the PDF",
        chunk_number=1,
    )
    chunks = [chunk, chunk2]
    print(chunks)
