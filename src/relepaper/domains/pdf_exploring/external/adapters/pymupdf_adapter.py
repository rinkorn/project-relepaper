# %%
# From https://gggauravgandhi.medium.com/handling-pdf-files-in-python-using-pymupdf-ba0b0b12ddc4

# %%
from pathlib import Path

import pymupdf

from relepaper.domains.pdf_exploring.interfaces import IPDFAdapter


class PyMuPDFAdapter(IPDFAdapter):
    def extract_metadata(self, pdf_path: Path) -> dict:
        doc = pymupdf.open(pdf_path)
        metadata = {**doc.metadata}
        metadata["page_count"] = doc.page_count
        return metadata

    def extract_table_of_contents(self, pdf_path: Path) -> dict:
        doc = pymupdf.open(pdf_path)
        table_of_contents = doc.get_toc()
        return table_of_contents

    def extract_text(self, pdf_path: Path) -> str:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page_number in range(doc.page_count):
            text += self.extract_page_text(pdf_path=pdf_path, page_number=page_number)
        return text

    def extract_images(self, pdf_path: Path) -> list[bytes]:
        doc = pymupdf.open(pdf_path)  # open a document
        out_images = []
        for page_index in range(len(doc)):  # iterate over pdf pages
            out_images.extend(self.extract_page_images(pdf_path=pdf_path, page_number=page_index))
        return out_images

    # def extract_page(self, pdf_path: Path, page_number: int) -> dict:
    #     doc = pymupdf.open(pdf_path)
    #     page = doc[page_number]
    #     return page

    def extract_page_text(self, pdf_path: Path, page_number: int) -> str:
        doc = pymupdf.open(pdf_path)
        page = doc[page_number]
        tp = page.get_textpage()
        return page.get_text(textpage=tp)

    def extract_page_images(self, pdf_path: Path, page_number: int) -> list[bytes]:
        doc = pymupdf.open(pdf_path)
        page = doc[page_number]
        images = page.get_images()
        out_images = []
        for image_index, img in enumerate(images, start=1):  # enumerate the image list
            xref = img[0]  # get the XREF of the image
            pix = pymupdf.Pixmap(doc, xref)  # create a Pixmap
            if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            out_images.append(pix.tobytes())
        return out_images


if __name__ == "__main__":
    import time

    from relepaper.config.constants import PROJECT_PATH

    pdf_path = PROJECT_PATH / "data" / "pdf" / "1912.01603v3.pdf"

    tic = time.time()

    adapter = PyMuPDFAdapter()
    metadata = adapter.extract_metadata(pdf_path=pdf_path)
    print(metadata)
    text = adapter.extract_text(pdf_path=pdf_path)
    print(text)
    images = adapter.extract_images(pdf_path=pdf_path)
    print(f"len(images): {len(images)}")
    # page = adapter.extract_page(pdf_path=pdf_path, page_number=0)
    # print(page)
    page_text = adapter.extract_page_text(pdf_path=pdf_path, page_number=0)
    print(page_text)
    page_images = adapter.extract_page_images(pdf_path=pdf_path, page_number=1)
    print(f"len(page_images): {len(page_images)}")

    toc = time.time()
    print(f"Time taken: {toc - tic} seconds")

# %%
