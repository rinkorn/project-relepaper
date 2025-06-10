# %%
import time
from pathlib import Path

from pdfminer.high_level import extract_text

from relepaper.domains.pdf_exploring.interfaces import IAdapter


# %%
class PDFMinerAdapter(IAdapter):
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def extract_metadata(self) -> dict:
        # TODO: Implement
        return {}

    def extract_text(self) -> str:
        return extract_text(self.pdf_path)

    def extract_table_of_contents(self) -> dict:
        # TODO: Implement
        return {}

    def extract_page(self, page_number: int) -> dict:
        # TODO: Implement
        return {}

    def extract_page_text(self, page_number: int) -> str:
        # TODO: Implement
        return ""

    def extract_page_images(self, page_number: int) -> list[bytes]:
        # TODO: Implement
        return []

    def extract_images(self) -> list[bytes]:
        # TODO: Implement
        return []


if __name__ == "__main__":
    from relepaper.config.constants import PROJECT_PATH

    pdf_path = PROJECT_PATH / "data" / "pdf" / "1912.01603v3.pdf"

    tic = time.time()

    adapter = PDFMinerAdapter(pdf_path=pdf_path)
    print(adapter.extract_text())

    toc = time.time()
    print(f"Time taken: {toc - tic} seconds")
