from pathlib import Path

from relepaper.domains.openalex.entities.openalex_pdf import OpenAlexPDF, PDFDownloadStrategy


class OpenAlexPDFCreateService:
    def create_pdf_entity(
        self,
        url: str,
        dirname: Path,
        filename: str = None,
    ) -> OpenAlexPDF:
        return OpenAlexPDF(
            url=url,
            dirname=dirname,
            filename=filename,
            strategy=PDFDownloadStrategy.NONE,
        )


# %%
if __name__ == "__main__":
    from relepaper.config import PROJECT_PATH
    from relepaper.domains.openalex.entities.openalex_pdf import OpenAlexPDF

    openalex_pdf = OpenAlexPDFCreateService().create_pdf_entity(
        url="https://www.pnas.org/doi/pdf/10.1073/pnas.0902281106", dirname=PROJECT_PATH / "data" / "pdf"
    )
    print(openalex_pdf)
