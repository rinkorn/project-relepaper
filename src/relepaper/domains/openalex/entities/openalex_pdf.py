from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class PDFDownloadStrategy(Enum):
    NONE = "none"
    REQUESTS = "requests"
    SELENIUM = "selenium"
    SELENIUM_STEALTH = "selenium_stealth"


@dataclass
class OpenAlexPDF:
    url: str = ""
    dirname: Path = Path.cwd()
    filename: str | None = None
    strategy: PDFDownloadStrategy = PDFDownloadStrategy.NONE

    @property
    def is_downloaded(self) -> bool:
        if self.filename is not None and isinstance(self.filename, str):
            return self.strategy != PDFDownloadStrategy.NONE and (self.dirname / self.filename).is_file()
        return False

    @property
    def is_file_exist(self) -> bool:
        if self.filename is not None and isinstance(self.filename, str):
            return (self.dirname / self.filename).is_file()
        return False
