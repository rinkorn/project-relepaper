# %%
from pathlib import Path
from pprint import pprint
from typing import List, TypedDict

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from loguru import logger

from relepaper.domains.langgraph.entities.pdf_content_chunk import PDFContentChunk
from relepaper.domains.langgraph.entities.pdf_content_splitter import PDFContentSplitter
from relepaper.domains.langgraph.entities.pdf_length import PDFLength
from relepaper.domains.langgraph.entities.pdf_metadata_extracted import PDFMetadataExtracted
from relepaper.domains.langgraph.interfaces import IWorkflowBuilder, IWorkflowEdge, IWorkflowNode
from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.pdf_exploring.entities.pdf_document import PDFDocument
from relepaper.domains.pdf_exploring.external.adapters.factory import AdapterFactory
from relepaper.domains.pdf_exploring.services.pdf_content_service import PDFDocumentService

__all__ = [
    "PDFAnalyserState",
    "PDFAnalyserWorkflowBuilder",
]


# %%
class PDFAnalyserState(TypedDict):
    pdf_document: PDFDocument
    pdf_metadata_extracted: PDFMetadataExtracted
    pdf_chunks: List[PDFContentChunk]
    pdf_chunks_metadata_extracted: List[PDFMetadataExtracted]
    short_long_pdf_length_threshold: int = 100000
    max_chunk_length: int = 100000
    max_chunks_count: int = 10
    intersection_length: int = 1000


def get_response_schemas() -> List[ResponseSchema]:
    return [
        ResponseSchema(
            name="title",
            description="Extract the title of the PDF file. ",
            type="string",
        ),
        ResponseSchema(
            name="abstract",
            description="Extract the abstract of the PDF file. min 500, max 10000 characters",
            type="string",
        ),
        ResponseSchema(
            name="keywords",
            description="Extract or tag the keywords of the PDF file. The keywords should be in the format of a list of strings. min 10, max 200",
            type="array",
            items={"type": "string"},
        ),
        ResponseSchema(
            name="authors",
            description="Extract the authors of the PDF file. min 1, max 20",
            type="array",
            items={"type": "string"},
        ),
        ResponseSchema(
            name="year",
            description="Extract the year of the PDF file. min 0, max 2099",
            type="number",
        ),
    ]


def get_prompt_template() -> str:
    return (
        "You are a helpful assistant that extracts the metadata from a PDF file. "
        "If the needed metadata is in the content, then just copy it. "
        "\n/no-think\n\n"
        "\nPDF METADATA:\n{pdf_metadata}\n\n"
        "\nPDF CONTENT:\n{pdf_content}\n\n "
        "\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
    )


class PDFMetadataExtractorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        pdf_document = state["pdf_document"]

        response_schemas = get_response_schemas()
        prompt_template = get_prompt_template()
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
            "pdf_metadata_extracted": PDFMetadataExtracted(
                title=response["title"],
                abstract=response["abstract"],
                keywords=response["keywords"],
                authors=response["authors"],
                year=response["year"],
            ),
        }
        lg.trace("end")
        return output


if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI

    from relepaper.domains.pdf_exploring.entities.pdf_document import PDFDocument, PDFMetadata, PDFText

    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.0,
    )

    pdf_document = PDFDocument(
        text=PDFText(text="Hello, world! This is a test of the PDFText class."),
        metadata=PDFMetadata(
            title="Test",
            authors=["John Doe"],
            keywords=["test", "test2"],
            year=2021,
        ),
    )
    node = PDFMetadataExtractorNode(llm=llm)
    output = node(state={"pdf_document": pdf_document})
    pprint(output)


# %%
class PDFLengthConditionalEdge(IWorkflowEdge):
    def __call__(self, state: PDFAnalyserState) -> PDFLength:
        logger.bind(classname=self.__class__.__name__).trace("start")
        pdf_document = state["pdf_document"]
        short_long_pdf_length_threshold = state["short_long_pdf_length_threshold"]
        logger.bind(classname=self.__class__.__name__).debug(f"Length of pdf_document: {len(pdf_document)}")
        logger.bind(classname=self.__class__.__name__).debug(f"Short_long_threshold: {short_long_pdf_length_threshold}")
        if len(pdf_document) > short_long_pdf_length_threshold:
            logger.bind(classname=self.__class__.__name__).trace("long_pdf: done")
            return PDFLength.LONG
        else:
            logger.bind(classname=self.__class__.__name__).trace("short_pdf: done")
            return PDFLength.SHORT


# %%
class PDFTextSplitterNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        logger.bind(classname=self.__class__.__name__).trace("start")
        pdf_document = state["pdf_document"]
        max_chunk_length = state["max_chunk_length"]
        max_chunks_count = state["max_chunks_count"]
        intersection_length = state["intersection_length"]
        splitter = PDFContentSplitter(
            pdf_document,
            max_chunk_length=max_chunk_length,
            max_chunks_count=max_chunks_count,
            intersection_length=intersection_length,
        )
        pdf_chunks = splitter.get_chunks()
        output = {
            "pdf_chunks": pdf_chunks,
        }
        logger.bind(classname=self.__class__.__name__).trace("end")
        return output


# %%
class PDFChunksMetadataExtractNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        logger.bind(classname=self.__class__.__name__).trace("start")
        pdf_document = state["pdf_document"]
        pdf_chunks = state["pdf_chunks"]
        pdf_chunks_metadata_extracted = []

        response_schemas = get_response_schemas()
        prompt_template = get_prompt_template()
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["pdf_content", "pdf_metadata"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | self._llm | output_parser

        for pdf_chunk in pdf_chunks:
            response = chain.invoke(
                {
                    "pdf_content": pdf_chunk.text,
                    "pdf_metadata": pdf_document.metadata,
                }
            )
            pdf_chunks_metadata_extracted.append(
                PDFMetadataExtracted(
                    title=response["title"],
                    abstract=response["abstract"],
                    keywords=response["keywords"],
                    authors=response["authors"],
                    year=response["year"],
                )
            )

        output = {
            "pdf_chunks_metadata_extracted": pdf_chunks_metadata_extracted,
        }
        logger.bind(classname=self.__class__.__name__).trace("end")
        return output


# %%
def get_prompt_template_for_union() -> str:
    return (
        "You are a helpful assistant that helps to unite the metadata from the chunks of the analyzed parts of the PDF file. "
        "Look at all the metadata in the sequential chunks of one PDF file and make conclusions about which metadata is the best for the full description of the PDF file. "
        "\nABSTRACT reformulate it to be a single, more understandable, more informative, combining and voluminous."
        "\nKEYWORDS you can enumerate, but do not duplicate. Do not invent new keywords. The most frequent and important keywords should be in the beginning of the list."
        "\nAUTHORS try to take from the first chunk. Do not invent new authors."
        "\nYEAR try to take from the first chunk. Do not invent new years. Do not unite chunks."
        "\nTITLE try to take from the first chunk. Do not invent new titles. Do not unite chunks."
        "\n/no-think\n\n"
        "\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
        "\nPDF CHUNKS METADATA EXTRACTED:\n{pdf_chunks_metadata_extracted}\n\n"
    )


class PDFChunksMetadataUnionNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        logger.bind(classname=self.__class__.__name__).trace("start")
        pdf_chunks_metadata_extracted = state["pdf_chunks_metadata_extracted"]
        if len(pdf_chunks_metadata_extracted) == 0:
            raise ValueError("pdf_chunks_metadata_extracted is empty")

        if len(pdf_chunks_metadata_extracted) == 1:
            pdf_metadata_extracted = pdf_chunks_metadata_extracted[0]
        else:
            prompt_template = get_prompt_template_for_union()
            response_schemas = get_response_schemas()
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["pdf_chunks_metadata_extracted"],
                partial_variables={"format_instructions": format_instructions},
            )
            chain = prompt | self._llm | output_parser

            response = chain.invoke(
                {
                    "pdf_chunks_metadata_extracted": pdf_chunks_metadata_extracted,
                }
            )
            pdf_metadata_extracted = PDFMetadataExtracted(
                title=response["title"],
                abstract=response["abstract"],
                keywords=response["keywords"],
                authors=response["authors"],
                year=response["year"],
            )

        output = {
            "pdf_metadata_extracted": pdf_metadata_extracted,
        }
        logger.bind(classname=self.__class__.__name__).trace("end")
        return output


if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.0,
    )
    node = PDFChunksMetadataUnionNode(llm=llm)
    output = node(
        state={
            "pdf_chunks_metadata_extracted": [
                PDFMetadataExtracted(
                    abstract="This chapter introduces a contrastive-curiosity-driven learning "
                    "framework (CCLF) to improve sample efficiency in reinforcement "
                    "learning. The framework integrates contrastive curiosity, "
                    "derived from the agent's internal belief about augmented "
                    "observations, into key components of RL such as experience "
                    "replay, training input selection, and Q-function regularization. "
                    "By focusing on under-explored samples and their augmentations, "
                    "CCLF enhances exploration while reducing computational overhead. "
                    "Experimental results demonstrate its effectiveness across "
                    "various benchmarks, including DeepMind Control (DMC), Atari "
                    "Games, and MiniGrid.",
                    authors=["Hen Li", "Zhiyuan Zhang", "Ying Nian Wu"],
                    keywords=[
                        "Curiosity-driven learning",
                        "Reinforcement Learning",
                        "Contrastive Learning",
                        "Sample Efficiency",
                        "Data Augmentation",
                    ],
                    title="A Contrastive-Curiosity-Driven Learning Framework for "
                    "Sample-Efficient Reinforcement Learning",
                    year=2030,
                ),
                PDFMetadataExtracted(
                    abstract="The proposed input selection process based on curiosity can be "
                    "seen as a selective adversary to the representation learning "
                    "process by the contrastive loss in Eq. 3.1. This is because the "
                    "agent curiously selects samples that are challenging for "
                    "matching, which maximizes the encoder loss L 𝑓, while at the "
                    "same time it improves the encoder with difficult inputs by "
                    "minimizing L 𝑓. As a result, the agent can learn an improved "
                    "encoder that is more robust to different views of observations "
                    "with less training required, compared to using random inputs.",
                    authors=["Author(s) not specified in the provided content."],
                    keywords=[
                        "Reinforcement Learning",
                        "Contrastive Curiosity",
                        "Sample Efficiency",
                        "Data Augmentation",
                        "Unsupervised Data Collection",
                    ],
                    title="A Contrastive-Curiosity-Driven Learning Framework for "
                    "Sample-Efficient Reinforcement Learning",
                    year=2023,
                ),
                PDFMetadataExtracted(
                    abstract="This chapter introduces a novel curiosity-driven unsupervised "
                    "data collection method known as CUDC, strategically designed to "
                    "enhance the quality of datasets within the multi-task offline RL "
                    "setting. The framework leverages hot cognition mechanisms to "
                    "enable self-adaptive evolution and incorporates collative "
                    "variables such as novelty, surprisingness, uncertainty, "
                    "complexity, and change to formulate an intrinsic reward for "
                    "curiosity-driven learning. Additionally, it proposes a data "
                    "selection mechanism that allows the model to focus on "
                    "informative data while maintaining stability and computational "
                    "efficiency. The results demonstrate that adapting the temporal "
                    "distance is crucial for collecting high-quality datasets, and "
                    "the proposed method outperforms existing approaches in terms of "
                    "sample efficiency and learning performance.",
                    authors=["Hongsheng Qian", "Chao Sun", "Chao Miao"],
                    keywords=[
                        "Curiosity-Driven Learning",
                        "Offline Reinforcement Learning",
                        "Adaptive Temporal Distances",
                        "Data Collection",
                        "Self-Adaptive Hot Cognition",
                    ],
                    title="A Curiosity-Driven Unsupervised Data Collection Method with "
                    "Adaptive Temporal Distances for Offline Reinforcement Learning",
                    year=2023,
                ),
                PDFMetadataExtracted(
                    abstract="The detailed setting of hyper-parameters is provided in Table A.3 and Table A.4.",
                    authors=["C. Sun", "H. Qian", "C. Miao"],
                    keywords=[
                        "Contrastive Learning",
                        "Reinforcement Learning",
                        "Sample Efficiency",
                        "Curiosity-driven Exploration",
                        "MiniGrid",
                    ],
                    title="CCLF: Contrastive Curiosity-Driven Learning for Sample-Efficient Reinforcement Learning",
                    year=2021,
                ),
            ]
        },
    )
    pprint(output)


# %%
class PDFAnalyserWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def build(self, **kwargs) -> StateGraph:
        logger.bind(classname=self.__class__.__name__).trace("start")
        graph = StateGraph(PDFAnalyserState)
        graph.add_node("pdf_metadata_extract", PDFMetadataExtractorNode(llm=self._llm))
        graph.add_node("pdf_text_splitter", PDFTextSplitterNode(llm=self._llm))
        graph.add_node("pdf_chunks_metadata_extract", PDFChunksMetadataExtractNode(llm=self._llm))
        graph.add_node("pdf_chunks_metadata_union", PDFChunksMetadataUnionNode(llm=self._llm))

        graph.add_conditional_edges(
            START,
            PDFLengthConditionalEdge(),
            {
                PDFLength.LONG: "pdf_text_splitter",
                PDFLength.SHORT: "pdf_metadata_extract",
            },
        )
        graph.add_edge("pdf_metadata_extract", END)
        graph.add_edge("pdf_text_splitter", "pdf_chunks_metadata_extract")
        graph.add_edge("pdf_chunks_metadata_extract", "pdf_chunks_metadata_union")
        graph.add_edge("pdf_chunks_metadata_union", END)
        compiled_graph = graph.compile()
        logger.bind(classname=self.__class__.__name__).trace("end")
        return compiled_graph


# %%
if __name__ == "__main__":
    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # llm = ChatOllama(
    #     model="hf.co/unsloth/Qwen3-30B-A3B-128K-GGUF:Q4_1",
    #     temperature=0.0,
    #     max_tokens=128000,
    # )
    from langchain.chat_models import ChatOpenAI

    from relepaper.domains.langgraph.workflows.utils.graph_displayer import (
        DisplayMethod,
        GraphDisplayer,
    )

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
        # OpenAlexPDF(
        #     url="https://www.mdpi.com/1996-1073/10/11/1846/pdf?version=1510484667",
        #     dirname=Path(
        #         "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
        #     ),
        #     filename="energies-10-01846.pdf",
        #     strategy=PDFDownloadStrategy.SELENIUM,
        # ),
        # OpenAlexPDF(
        #     url="https://some-url.com",
        #     dirname=Path(
        #         "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
        #     ),
        #     filename="3219819.3220096.pdf",
        #     strategy=PDFDownloadStrategy.SELENIUM,
        # ),
        # OpenAlexPDF(
        #     url="https://some-url.com",
        #     dirname=Path(
        #         "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
        #     ),
        #     filename="Intl J Robust   Nonlinear - 2021 - Wan - Optimal control and learning for cyber‐physical systems.pdf",
        #     strategy=PDFDownloadStrategy.SELENIUM,
        # ),
        OpenAlexPDF(
            url="https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-021-00561-9",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="s13321-021-00561-9.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        # OpenAlexPDF(
        #     url="https://dr.ntu.edu.sg/bitstream/10356/172831/2/main_thesis.pdf",
        #     dirname=Path(
        #         "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
        #     ),
        #     filename="main_thesis.pdf",
        #     strategy=PDFDownloadStrategy.SELENIUM,
        # ),
    ]
    pdf_adapter = AdapterFactory.create("pymupdf")
    pdf_service = PDFDocumentService(pdf_adapter=pdf_adapter)

    pdf_documents = []
    for pdf in pdfs:
        pdf_document = pdf_service.load_pdf_document(
            pdf.dirname / pdf.filename,
            # max_text_length=200000,
        )
        pdf_documents.append(pdf_document)

    workflow = PDFAnalyserWorkflowBuilder(llm=llm).build()

    displayer = GraphDisplayer(workflow).set_strategy(DisplayMethod.MERMAID)
    displayer.display()

    for pdf_document in pdf_documents:
        state_start = {
            "pdf_document": pdf_document,
            "pdf_metadata_extracted": None,
            "pdf_chunks": [],
            "pdf_chunks_metadata_extracted": [],
            "short_long_pdf_length_threshold": 100000,
            "max_chunk_length": 100000,
            "max_chunks_count": 10,
            "intersection_length": 2000,
        }
        state_end = workflow.invoke(input=state_start)
        pprint(state_end["pdf_metadata_extracted"])
        print(f"Length of pdf_document: {len(state_end['pdf_document'])}")
        print(f"Short_long_pdf_length_threshold: {state_end['short_long_pdf_length_threshold']}")
        print(f"Chunks count: {len(state_end['pdf_chunks'])}")
        print(f"Intersection_length: {state_end['intersection_length']}")
        print(f"Max_chunk_length: {state_end['max_chunk_length']}")
        print(f"Max_chunks_count: {state_end['max_chunks_count']}")
        print("-" * 100)


# %%
