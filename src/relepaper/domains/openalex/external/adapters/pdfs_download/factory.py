# %%
from loguru import logger

from relepaper.domains.openalex.entities.pdf import PDFDownloadStrategy
from relepaper.domains.openalex.external.adapters.pdfs_download.request_adapter import RequestsPDFDownloadAdapter
from relepaper.domains.openalex.external.adapters.pdfs_download.selenium_adapter import SeleniumPDFDownloadAdapter
from relepaper.domains.openalex.external.adapters.pdfs_download.selenium_stealth_adapter import (
    SeleniumStealthPDFDownloadAdapter,
)
from relepaper.domains.openalex.external.interfaces import IAdapter, IAdapterFactory


# %%
class PDFDownloadAdapterFactory(IAdapterFactory):
    @staticmethod
    def create(strategy: PDFDownloadStrategy = PDFDownloadStrategy("requests")) -> IAdapter:
        match strategy:
            case PDFDownloadStrategy.REQUESTS:
                return RequestsPDFDownloadAdapter()
            case PDFDownloadStrategy.SELENIUM:
                return SeleniumPDFDownloadAdapter()
            case PDFDownloadStrategy.SELENIUM_STEALTH:
                return SeleniumStealthPDFDownloadAdapter()
            case _:
                logger.error(f"Invalid strategy: {strategy}")
                raise ValueError(f"Invalid strategy: {strategy}")


# %%
if __name__ == "__main__":
    from relepaper.config.dev_settings import get_dev_settings
    from relepaper.domains.openalex.entities.pdf import OpenAlexPDF

    openalex_pdf = OpenAlexPDF(
        url="https://arxiv.org/pdf/1912.01603",
        # url="https://www.google.com/pdf1",
        dirname=get_dev_settings().project_path / "data" / "pdf",
    )

    service = PDFDownloadAdapterFactory.create(strategy=PDFDownloadStrategy("requests"))
    # service.download_pdf(openalex_pdf)

    service = PDFDownloadAdapterFactory.create(strategy=PDFDownloadStrategy("selenium"))
    # service.download_pdf(openalex_pdf)

    service = PDFDownloadAdapterFactory.create(strategy=PDFDownloadStrategy("selenium_stealth"))
    service.download(openalex_pdf)

    print(openalex_pdf.__dict__)
    print(f"exist: {openalex_pdf.is_file_exist}")
    print(f"downloaded: {openalex_pdf.is_downloaded}")
