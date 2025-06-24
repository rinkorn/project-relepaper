import logging
from pathlib import Path
from pprint import pprint
from typing import List

from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.openalex.entities.work import OpenAlexWork
from relepaper.domains.openalex.external.adapters.pdfs_download.factory import PDFDownloadAdapterFactory
from relepaper.domains.openalex.external.interfaces import IAdapter
from relepaper.domains.openalex.services.interfaces import IService

logger = logging.getLogger(__name__)


class OpenAlexPdfDownloadService(IService):
    def __init__(
        self,
        strategy: PDFDownloadStrategy = PDFDownloadStrategy("selenium"),
        dirname: Path = Path().cwd() / "data" / "pdf",
    ):
        self._strategy = strategy
        self._dirname = dirname
        self._adapter: IAdapter = PDFDownloadAdapterFactory.create(strategy=self._strategy)

    def download_from_work(
        self,
        work: OpenAlexWork,
        timeout: int = 60,
    ) -> OpenAlexPDF:
        """Download a PDF from the internet.

        Args:
            work: OpenAlexWork object
            dirname: directory name
            strategy: strategy to download pdf ('requests', 'selenium', 'selenium_stealth')
            timeout: timeout for download
        Returns:
            OpenAlexPDF object
        """
        openalex_pdf = OpenAlexPDF(
            url=work.pdf_url,
            dirname=self._dirname,
            strategy=self._strategy,
        )
        if not openalex_pdf:
            logger.error(f"OpenAlexPDF is not valid: {openalex_pdf}")
            return None
        self._adapter.download(
            openalex_pdf,
            timeout=timeout,
        )
        return openalex_pdf

    def download_from_url(
        self,
        url: str,
        timeout: int = 60,
    ) -> OpenAlexPDF:
        """Download a PDF from the internet.

        Args:
            url: URL of the PDF
        Returns:
            OpenAlexPDF object
        """

        openalex_pdf = OpenAlexPDF(
            url=url,
            dirname=self._dirname,
            strategy=self._strategy,
        )
        if not openalex_pdf:
            logger.error(f"OpenAlexPDF is not valid: {openalex_pdf}")
            return None
        self._adapter.download(
            openalex_pdf,
            timeout=timeout,
        )
        return openalex_pdf

    def download_from_works(
        self,
        works: List[OpenAlexWork],
        timeout: int = 60,
    ) -> List[OpenAlexPDF]:
        """Download all PDFs from the internet.

        Args:
            works: List of OpenAlexWork objects
            dirname: directory name
            strategy: strategy to download pdf ('requests', 'selenium', 'selenium_stealth')
            timeout: timeout for download
        Returns:
            List of OpenAlexPDF objects
        """
        return [self.download_from_work(work, timeout) for work in works]

    def download_from_urls(
        self,
        urls: List[str],
        timeout: int = 60,
    ) -> List[OpenAlexPDF]:
        """Download all PDFs from the internet.

        Args:
            urls: List of URLs
        Returns:
            List of OpenAlexPDF objects
        """
        return [self.download_from_url(url, timeout) for url in urls]


# from concurrent.futures import ThreadPoolExecutor

# def batch_chain(inputs: list) -> list:
#     with ThreadPoolExecutor(max_workers=3) as executor:
#         return list(executor.map(chain.invoke, inputs))

# out = batch_chain(entries)

if __name__ == "__main__":
    works = [
        {
            "id": "http://openalex.org/W12345",
            "title": "Some title",
            "authors": [
                {
                    "id": "http://openalex.org/A12345",
                    "name": "Some author",
                }
            ],
            "primary_location": {
                "pdf_url": "https://www.google.com/pdf1",
            },
        },
        {
            "id": "http://openalex.org/W12346",
            "title": "Another title",
            "authors": [
                {
                    "id": "http://openalex.org/A123456",
                    "name": "Another author",
                }
            ],
            "primary_location": {
                "pdf_url": "https://www.google.com/pdf2",
            },
        },
    ]
    works = [OpenAlexWork.from_dict(work) for work in works]
    pprint(works)
    download_service = OpenAlexPdfDownloadService(
        strategy=PDFDownloadStrategy("selenium"),
        dirname=Path("test"),
    )
    download_service.download_from_works(works, timeout=30)


# %%
if __name__ == "__main__":
    # url = "https://www.google.com/pdf2"
    download_service = OpenAlexPdfDownloadService(
        strategy=PDFDownloadStrategy("selenium"),
        dirname=Path("test"),
    )
    url = None
    openalex_pdf = download_service.download_from_url(url, timeout=30)
    print(openalex_pdf)
