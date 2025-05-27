# %%
import logging

from relepaper.domains.openalex.services.interfaces import IService
from relepaper.domains.openalex.services.pdf_download.pdf_requests_service import RequestsPDFDownloadService
from relepaper.domains.openalex.services.pdf_download.pdf_selenium_service import SeleniumPDFDownloadService
from relepaper.domains.openalex.services.pdf_download.pdf_selenium_stealth_service import (
    SeleniumStealthPDFDownloadService,
)

logger = logging.getLogger(__name__)


# %%
class PDFDownloadServiceFactory:
    @staticmethod
    def create(strategy: str = "requests") -> IService:
        match strategy:
            case "requests":
                return RequestsPDFDownloadService()
            case "selenium":
                return SeleniumPDFDownloadService()
            case "selenium_stealth":
                return SeleniumStealthPDFDownloadService()
            case _:
                logger.error(f"Invalid strategy: {strategy}")
                raise ValueError(f"Invalid strategy: {strategy}")


# %%
if __name__ == "__main__":
    from relepaper.config.dev_settings import get_dev_settings
    from relepaper.domains.openalex.entities.openalex_pdf import OpenAlexPDF

    openalex_pdf = OpenAlexPDF(
        url="https://arxiv.org/pdf/1912.01603",
        dirname=get_dev_settings().project_path / "data" / "pdf",
    )

    # service = PDFDownloadServiceFactory.create(strategy="requests")
    # service.download_pdf(openalex_pdf)

    # service = PDFDownloadServiceFactory.create(strategy="selenium")
    # service.download_pdf(openalex_pdf)

    service = PDFDownloadServiceFactory.create(strategy="selenium_stealth")
    service.download(openalex_pdf)

    print(openalex_pdf.__dict__)
    print(f"exist: {openalex_pdf.is_file_exist}")
    print(f"downloaded: {openalex_pdf.is_downloaded}")
