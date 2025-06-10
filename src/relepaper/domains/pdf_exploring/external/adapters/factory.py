# %%
from relepaper.domains.pdf_exploring.external.adapters.pdfminer_adapter import PDFMinerAdapter
from relepaper.domains.pdf_exploring.external.adapters.pymupdf_adapter import PyMuPDFAdapter
from relepaper.domains.pdf_exploring.interfaces import IAdapter


class AdapterFactory:
    @staticmethod
    def create(adapter_name: str) -> IAdapter:
        if adapter_name == "pdfminer":
            return PDFMinerAdapter
        elif adapter_name == "pymupdf":
            return PyMuPDFAdapter
        else:
            raise ValueError(f"Adapter {adapter_name} not found")


# %%
if __name__ == "__main__":
    import time
    from relepaper.config.constants import PROJECT_PATH

    pdf_path = PROJECT_PATH / "data" / "pdf" / "1912.01603v3.pdf"

    tic = time.time()
    print("PDFMinerAdapter")
    adapter = AdapterFactory.create("pdfminer")(pdf_path=pdf_path)
    adapter.extract_text()
    toc = time.time()
    print(f"Time taken: {toc - tic} seconds")

    tic = time.time()
    print("PyMuPDFAdapter")
    adapter = AdapterFactory.create("pymupdf")(pdf_path=pdf_path)
    adapter.extract_text()
    toc = time.time()
    print(f"Time taken: {toc - tic} seconds")
