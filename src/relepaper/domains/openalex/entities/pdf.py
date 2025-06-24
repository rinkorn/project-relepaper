from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class PDFDownloadStrategy(Enum):
    NONE = "none"
    REQUESTS = "requests"
    SELENIUM = "selenium"
    SELENIUM_STEALTH = "selenium_stealth"


@dataclass
class OpenAlexPDF:
    url: str | None = None
    dirname: Path | None = None
    filename: str | None = None
    strategy: PDFDownloadStrategy | None = None

    # Новые поля для связи с запросами
    source_query: Optional[str] = None
    source_work_id: Optional[str] = None
    source_query_index: Optional[int] = None

    def __bool__(self) -> bool:
        return bool(self.url)

    def __post_init__(self):
        if self.url is None:
            self.url = ""
        if self.dirname is None:
            self.dirname = Path.cwd()
        if self.filename is None:
            self.filename = ""
        if self.strategy is None:
            self.strategy = PDFDownloadStrategy.NONE

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

    @property
    def file_path(self) -> Path:
        if self.filename is not None and isinstance(self.filename, str):
            return self.dirname / self.filename
        return Path.cwd()
