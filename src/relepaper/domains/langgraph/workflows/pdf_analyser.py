# %%
import logging
from pathlib import Path
from pprint import pprint

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.document_loaders.pdf import (
    PyMuPDFLoader,
    # UnstructuredPDFLoader,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy

# __all__ = [
#     "PDFContentReaderNode",
#     "PDFContentAnalyzerNode",
# ]

# %%
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    formatter = logging.Formatter("%(message)s")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# %%
if __name__ == "__main__":
    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # llm = ChatOllama(
    #     model="hf.co/unsloth/Qwen3-30B-A3B-128K-GGUF:Q4_1",
    #     temperature=0.0,
    #     max_tokens=128000,
    # )
    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=1.0,
    )


# %%
# pypdf
# pymupdf - fastests
# pypdfium2
# pdfplumber
# unstructured
def read_pdf_content(pdf_path: Path, num_pages: int = 10) -> tuple[str, dict]:
    """
    Read the content of a PDF file.

    Args:
        pdf: OpenAlexPDF

    Returns:
        PDFDocument
    """

    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()
    page_contents = []
    metadata = {}
    for page in pages[:num_pages]:
        page_contents.append(page.page_content)
        metadata.update(page.metadata)
    content = "\n".join(page_contents)
    return content, metadata


def extract_metadata_from_pdf(pdf_content: str, pdf_metadata: dict, llm: BaseChatModel) -> dict:
    response_schemas = [
        ResponseSchema(
            name="title",
            description="Copy the title of the PDF file.",
            type="string",
        ),
        ResponseSchema(
            name="abstract",
            description="Copy the abstract of the PDF file.",
            type="string",
        ),
        ResponseSchema(
            name="keywords",
            description="Copy the keywords of the PDF file.",
            type="array",
            items={"type": "string"},
            minItems=1,
            maxItems=20,
        ),
        ResponseSchema(
            name="authors",
            description="Copy the authors of the PDF file.",
            type="string",
        ),
        ResponseSchema(
            name="year",
            description="Copy the year of the PDF file.",
            type="number",
            minValue=2000,
            maxValue=2025,
            multipleOf=1,
        ),
        ResponseSchema(
            name="journal",
            description="Copy the journal of the PDF file.",
            type="string",
        ),
    ]

    prompt_template = (
        "You are a helpful assistant that extracts the metadata from a PDF file. "
        "If the needed metadata is in the content, then just copy it. "
        "\n/no-think\n\n"
        "\nPDF METADATA:\n{pdf_metadata}\n\n"
        "\nPDF CONTENT:\n{pdf_content}\n\n "
        "\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
    )
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["pdf_content", "pdf_metadata"],
        partial_variables={"format_instructions": format_instructions},
    )
    chain = prompt | llm | output_parser

    response = chain.invoke(
        {
            "pdf_content": pdf_content,
            "pdf_metadata": pdf_metadata,
        }
    )
    output = {
        "title": response["title"],
        "authors": response["authors"],
        "year": response["year"],
        "journal": response["journal"],
        "keywords": response["keywords"],
        "abstract": response["abstract"],
    }
    return output


if __name__ == "__main__":
    pdfs = [
        OpenAlexPDF(
            url="https://www.mdpi.com/1996-1073/10/11/1846/pdf?version=1510484667",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="energies-10-01846.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-021-00561-9",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="s13321-021-00561-9.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="3219819.3220096.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="Intl J Robust   Nonlinear - 2021 - Wan - Optimal control and learning for cyber‐physical systems.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
    ]

    pdf_data = [read_pdf_content(pdf.dirname / pdf.filename, num_pages=3) for pdf in pdfs]
    contents = [content for content, _ in pdf_data]
    metadata = [metadata for _, metadata in pdf_data]

    results = []
    for content, metadata in zip(contents, metadata):
        extracted_metadata = extract_metadata_from_pdf(content, metadata, llm)
        pprint(extracted_metadata)
        results.append(extracted_metadata)
        print("-" * 100)

    # extracted_metadata = [
    #     extract_metadata_from_pdf(pdf[0], llm) for pdf in pdf_contents
    # ]
    # pprint(extracted_metadata)


# %%
