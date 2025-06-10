# %%
from pathlib import Path

from relepaper.domains.pdf_exploring.entities.pdf_document import PDFDocument, PDFMetadata, PDFText
from relepaper.domains.pdf_exploring.interfaces import IAdapter, IService


class PDFDocumentService(IService):
    def __init__(self, pdf_adapter: IAdapter):
        self.pdf_adapter = pdf_adapter

    def create_pdf_document(self, pdf_path: Path) -> PDFDocument:
        """Read the content of a PDF file."""

        pdf_metadata = self.pdf_adapter.extract_metadata()
        pdf_text = self.pdf_adapter.extract_text()

        pdf_metadata = PDFMetadata(
            num_pages=pdf_metadata.num_pages,
            title=pdf_metadata.title,
            author=pdf_metadata.author,
            subject=pdf_metadata.subject,
            keywords=pdf_metadata.keywords,
            creator=pdf_metadata.creator,
            producer=pdf_metadata.producer,
            creationDate=pdf_metadata.creationDate,
            modDate=pdf_metadata.modDate,
            trapped=pdf_metadata.trapped,
            encryption=pdf_metadata.encryption,
        )

        pdf_text = PDFText(text=pdf_text)

        pdf_document: PDFDocument = PDFDocument(
            text=pdf_text,
            metadata=pdf_metadata,
        )
        return pdf_document


# %%
if __name__ == "__main__":
    from relepaper.config.constants import PROJECT_PATH
    from relepaper.domains.pdf_exploring.external.adapters.factory import AdapterFactory

    pdf_path = PROJECT_PATH / "data" / "pdf" / "1912.01603v3.pdf"
    pdf_adapter = AdapterFactory.create("pdfminer")(pdf_path=pdf_path)
    pdf_service = PDFDocumentService(pdf_adapter=pdf_adapter)
    pdf_document = pdf_service.create_pdf_document(pdf_path=pdf_path)
    print(pdf_document)


# %%
