# %%
from pathlib import Path

from relepaper.domains.pdf_exploring.entities.pdf_document import PDFDocument, PDFMetadata, PDFText
from relepaper.domains.pdf_exploring.interfaces import IPDFAdapter, IService


class PDFDocumentService(IService):
    def __init__(self, pdf_adapter: IPDFAdapter):
        self.pdf_adapter = pdf_adapter

    def load_pdf_document(self, pdf_path: Path) -> PDFDocument:
        """Read the content of a PDF file."""

        metadata = self.pdf_adapter.extract_metadata(pdf_path=pdf_path)
        text = self.pdf_adapter.extract_text(pdf_path=pdf_path)

        pdf_metadata = PDFMetadata(
            title=metadata.get("title", None),
            authors=metadata.get("authors", []),
            year=metadata.get("year", None),
            keywords=metadata.get("keywords", []),
            abstract=metadata.get("abstract", None),
            num_pages=metadata.get("page_count", None),
        )

        pdf_text = PDFText(text=text)
        # pdf_images = self.pdf_adapter.extract_images(pdf_path=pdf_path)

        pdf_document: PDFDocument = PDFDocument(
            text=pdf_text,
            metadata=pdf_metadata,
            # images=pdf_images,
        )
        return pdf_document


# %%
if __name__ == "__main__":
    from relepaper.config.constants import PROJECT_PATH
    from relepaper.domains.pdf_exploring.external.adapters.factory import AdapterFactory

    pdf_path = PROJECT_PATH / "data" / "pdf" / "1912.01603v3.pdf"
    adapter = AdapterFactory.create("pymupdf")

    pdf_metadata = adapter.extract_metadata(pdf_path=pdf_path)
    print(pdf_metadata)
    pdf_text = adapter.extract_text(pdf_path=pdf_path)
    print(pdf_text)
    pdf_images = adapter.extract_images(pdf_path=pdf_path)
    print(len(pdf_images))

    pdf_service = PDFDocumentService(pdf_adapter=adapter)
    pdf_document = pdf_service.load_pdf_document(pdf_path=pdf_path)
    # print(pdf_document.text)
    print(pdf_document.metadata)


# %%
if __name__ == "__main__":
    pdf_path = Path(
        "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
    )
    # pdf_path = pdf_path / "energies-10-01846.pdf"
    # pdf_path = pdf_path / "s13321-021-00561-9.pdf"
    # pdf_path = pdf_path / "3219819.3220096.pdf"
    # pdf_path = (
    #     pdf_path
    #     / "Intl J Robust   Nonlinear - 2021 - Wan - Optimal control and learning for cyber‐physical systems.pdf"
    # )

    pdf_adapter = AdapterFactory.create("pymupdf")
    pdf_service = PDFDocumentService(pdf_adapter=pdf_adapter)
    pdf_document = pdf_service.load_pdf_document(pdf_path=pdf_path)
    # print(pdf_document)
    print(pdf_document.text)
    print(pdf_document.metadata)

# %%
