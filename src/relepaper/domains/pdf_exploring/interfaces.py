from abc import ABC
from pathlib import Path


class IService(ABC):
    pass


class IPDFAdapter(ABC):
    def extract_metadata(self, pdf_path: Path) -> dict:
        raise NotImplementedError

    def extract_text(self, pdf_path: Path) -> str:
        raise NotImplementedError

    def extract_images(self, pdf_path: Path) -> list[bytes]:
        raise NotImplementedError

    def extract_page_text(self, pdf_path: Path, page_number: int) -> str:
        raise NotImplementedError

    def extract_page_images(self, pdf_path: Path, page_number: int) -> list[bytes]:
        raise NotImplementedError
