# %%
from pathlib import Path

from pdfminer.high_level import extract_text

from relepaper.domains.pdf_exploring.interfaces import IPDFAdapter


# %%
class PDFMinerAdapter(IPDFAdapter):
    def __init__(self):
        pass

    def extract_metadata(self, pdf_path: Path) -> dict:
        pass

    def extract_table_of_contents(self, pdf_path: Path) -> dict:
        pass

    def extract_text(self, pdf_path: Path) -> str:
        return extract_text(pdf_path)

    def extract_images(self, pdf_path: Path) -> list[bytes]:
        pass

    def extract_page_text(self, pdf_path: Path, page_number: int) -> str:
        pass

    def extract_page_images(self, pdf_path: Path, page_number: int) -> list[bytes]:
        pass


# %%
if __name__ == "__main__":
    from relepaper.config.constants import PROJECT_PATH

    pdf_path = PROJECT_PATH / "data" / "pdf" / "1912.01603v3.pdf"

    adapter = PDFMinerAdapter()
    print(adapter.extract_text(pdf_path=pdf_path))
