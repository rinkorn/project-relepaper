# %%
import logging
from pathlib import Path
from pprint import pprint
from typing import List, TypedDict

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from relepaper.domains.langgraph.workflows.interfaces import IWorkflowBuilder, IWorkflowNode
from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.pdf_exploring.entities.pdf_document import PDFDocument
from relepaper.domains.pdf_exploring.external.adapters.factory import AdapterFactory
from relepaper.domains.pdf_exploring.services.pdf_content_service import PDFDocumentService

__all__ = [
    "PDFMetadataExtractorState",
    "PDFMetadataExtractorWorkflowBuilder",
]

# %%
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    formatter = logging.Formatter("__log__: %(message)s")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# %%
class PDFMetadataExtractorState(TypedDict):
    pdf_document: PDFDocument
    extracted_title: str
    extracted_abstract: str
    extracted_keywords: List[str]
    extracted_year: int
    extracted_journal: str
    extracted_authors: List[str]


class PDFMetadataExtractorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: PDFMetadataExtractorState) -> PDFMetadataExtractorState:
        pdf_document = state["pdf_document"]

        response_schemas = [
            ResponseSchema(
                name="title",
                description="Extract the title of the PDF file.",
                type="string",
            ),
            ResponseSchema(
                name="abstract",
                description="Extract the abstract of the PDF file.",
                type="string",
            ),
            ResponseSchema(
                name="keywords",
                description="Extract or tag the keywords of the PDF file. The keywords should be in the format of a list of strings.",
                type="array",
                items={"type": "string"},
                minItems=1,
                maxItems=20,
            ),
            ResponseSchema(
                name="authors",
                description="Extract the authors of the PDF file.",
                type="array",
                items={"type": "string"},
                minItems=1,
                maxItems=20,
            ),
            ResponseSchema(
                name="year",
                description="Extract the year of the PDF file.",
                type="number",
                minValue=0,
                maxValue=2025,
                multipleOf=1,
            ),
            ResponseSchema(
                name="journal",
                description="Extract the journal of the PDF file.",
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
        chain = prompt | self._llm | output_parser

        response = chain.invoke(
            {
                "pdf_content": pdf_document.text,
                "pdf_metadata": pdf_document.metadata,
            }
        )
        output = {
            "extracted_title": response["title"],
            "extracted_authors": response["authors"],
            "extracted_year": response["year"],
            "extracted_journal": response["journal"],
            "extracted_keywords": response["keywords"],
            "extracted_abstract": response["abstract"],
        }
        return output


# if __name__ == "__main__":
#     from relepaper.domains.pdf_exploring.entities.pdf_document import PDFDocument, PDFMetadata, PDFText

#     llm = ChatOpenAI(
#         base_url="http://localhost:7007/v1",
#         api_key="not_needed",
#         temperature=0.0,
#     )

#     pdf_document = PDFDocument(
#         text=PDFText(text="Hello, world! This is a test of the PDFText class."),
#         metadata=PDFMetadata(
#             title="Test",
#             authors=["John Doe"],
#             keywords=["test", "test2"],
#         ),
#     )
#     node = PDFMetadataExtractorNode(llm=llm)
#     output = node(state={"pdf_document": pdf_document})
#     pprint(output)


# %%
class PDFMetadataExtractorWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def build(self, **kwargs) -> StateGraph:
        graph = StateGraph(PDFMetadataExtractorState)
        graph.add_node("pdf_metadata_extract", PDFMetadataExtractorNode(llm=self._llm))
        graph.add_edge(START, "pdf_metadata_extract")
        graph.add_edge("pdf_metadata_extract", END)
        return graph.compile()


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
        temperature=0.00,
    )
    # from langchain.chat_models.gigachat import GigaChat
    # # auth = os.getenv("GIGACHAT_CREDENTIALS")
    # llm = GigaChat(
    #     credentials="ZThkYmEyZWQtNjdjMi00NjBmLWE4MmUtMTM0NzRjNTZmMDM2OjY2YTMxMzA1LTlhNTQtNDNhOS05ZjIwLTgyMmMzOGIyODM5Nw==",
    #     model="GigaChat:max",  # GigaChat:max, GigaChat:lite, GigaChat:pro
    #     scope="GIGACHAT_API_CORP",
    #     verify_ssl_certs=False,
    #     profanity_check=False,
    # )
    # llm.invoke("Hello, world!")

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
        OpenAlexPDF(
            url="https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-021-00561-9",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="s13321-021-00561-9.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
    ]
    pdf_adapter = AdapterFactory.create("pymupdf")
    pdf_service = PDFDocumentService(pdf_adapter=pdf_adapter)

    pdf_documents = [pdf_service.load_pdf_document(pdf.dirname / pdf.filename) for pdf in pdfs]

    workflow = PDFMetadataExtractorWorkflowBuilder(llm=llm).build()
    for pdf_document in pdf_documents:
        state_start = {"pdf_document": pdf_document}
        state_end = workflow.invoke(input=state_start)
        pprint(state_end)
        print("-" * 100)


# %%
