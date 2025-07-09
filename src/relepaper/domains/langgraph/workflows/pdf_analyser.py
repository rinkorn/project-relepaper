# %%
import uuid
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import List, TypedDict

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from loguru import logger

from relepaper.domains.langgraph.entities.pdf_content_chunk import PDFContentChunk
from relepaper.domains.langgraph.entities.pdf_content_splitter import PDFContentSplitter
from relepaper.domains.langgraph.entities.pdf_length import PDFLength
from relepaper.domains.langgraph.entities.pdf_metadata_extracted import PDFMetadataExtracted
from relepaper.domains.langgraph.interfaces import IStrategy, IWorkflowBuilder, IWorkflowEdge, IWorkflowNode
from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.pdf_exploring.entities.pdf_document import PDFDocument, PDFMetadata, PDFText
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


# %%
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
            description="Extract or tag the keywords of the PDF file. The keywords should be in the format of a list of strings. min 0, max 100",
            type="array",
            items={"type": "string"},
        ),
        ResponseSchema(
            name="authors",
            description="Extract the authors of the PDF file. min 0, max 20",
            type="array",
            items={"type": "string"},
        ),
        ResponseSchema(
            name="year",
            description="Extract the year of the PDF file. min -1, max 2100",
            type="number",
        ),
    ]


# %%
def get_prompt_template(
    think: bool = False,
    pdf_content: str | None = None,
    pdf_metadata: str | None = None,
    format_instructions: str | None = None,
) -> str:
    prompt = (
        "You are a helpful assistant that extracts the metadata from a PDF file. "
        "If the needed metadata is in the content, then just copy it. "
    )
    if not think:
        prompt += "\n/no-think\n\n"
    if pdf_content:
        prompt += f"\nPDF CONTENT:\n{pdf_content}\n\n "
    if pdf_metadata:
        prompt += f"\nPDF METADATA:\n{pdf_metadata}\n\n"
    if format_instructions:
        prompt += f"\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
    return prompt


class ResponseSchemasMetadataExtractStrategy(IStrategy):
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
    import os

    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.0,
        num_predict=32000,
    )

    # from langchain.chat_models import ChatOpenAI
    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.0,
    # )

    pdf_document = PDFDocument(
        text=PDFText(
            text=(
                "DrugEx v2: de novo design of drug"
                "molecules by Pareto-based multi-objective "
                "reinforcement learning in polypharmacology"
                "Xuhan Liu1 , Kai Ye2 , Herman W. T. van Vlijmen1,3 , Michael T. M. Emmerich4 , Adriaan P. IJzerman1  and "
                "Gerard J. P. van Westen1*  "
                "Abstract In polypharmacology drugs are required to bind to multiple specific targets, for example to enhance efficacy or "
                "to reduce resistance formation. Although deep learning has achieved a breakthrough in de novo design in drug "
                "discovery, most of its applications only focus on a single drug target to generate drug-like active molecules. However, "
                "in reality drug molecules often interact with more than one target which can have desired (polypharmacology) or "
                "undesired (toxicity) effects. In a previous study we proposed a new method named DrugEx that integrates an explora-"
                "tion strategy into RNN-based reinforcement learning to improve the diversity of the generated molecules. Here, we "
                "extended our DrugEx algorithm with multi-objective optimization to generate drug-like molecules towards multiple "
                "targets or one specific target while avoiding off-targets (the two adenosine receptors,  A1AR and  A2AAR, and the potas-"
                "sium ion channel hERG in this study). In our model, we applied an RNN as the agent and machine learning predictors "
                "as the environment. Both the agent and the environment were pre-trained in advance and then interplayed under a "
                "reinforcement learning framework. The concept of evolutionary algorithms was merged into our method such that "
                "crossover and mutation operations were implemented by the same deep learning model as the agent. During the "
                "training loop, the agent generates a batch of SMILES-based molecules. Subsequently scores for all objectives provided "
                "by the environment are used to construct Pareto ranks of the generated molecules. For this ranking a non-dominated "
                "sorting algorithm and a Tanimoto-based crowding distance algorithm using chemical fingerprints are applied. Here, "
                "we adopted GPU acceleration to speed up the process of Pareto optimization. The final reward of each molecule is "
                "calculated based on the Pareto ranking with the ranking selection algorithm. The agent is trained under the guidance "
                "of the reward to make sure it can generate desired molecules after convergence of the training process. All in all we "
                "demonstrate generation of compounds with a diverse predicted selectivity profile towards multiple targets, offering "
                "the potential of high efficacy and low toxicity."
                "Keywords:  Deep learning, Adenosine receptors, Cheminformatics, Reinforcement learning, Multi-objective "
                "optimization, Exploration strategy"
            )
        ),
        metadata=PDFMetadata(
            title="Test",
            authors=["John Doe"],
            keywords=["test", "test2"],
            year=2019,
        ),
    )
    node = ResponseSchemasMetadataExtractStrategy(llm=llm)
    output = node(state={"pdf_document": pdf_document})
    pprint(output)


# %%
def get_prompt_template_for_structured_output(
    pdf_metadata: str | None = None,
    pdf_content: str | None = None,
    think: bool = False,
):
    prompt = (
        "You are a helpful assistant that extracts the metadata from a PDF file. "
        "If the needed metadata is in the content, then just copy it. "
        "If the needed metadata is not in the content, DO NOT FILL THE FIELDS, just return empty fields. "
        "Generate title from the content[if it is not related in the content]. "
        "Generate keywords from the content. "
    )
    if not think:
        prompt += "\n/no-think\n\n"
    if pdf_content:
        prompt += f"\nPDF CONTENT:\n{pdf_content}\n\n "
    if pdf_metadata:
        prompt += f"\nPDF METADATA:\n{pdf_metadata}\n\n"
    return prompt


# def get_structured_output_schema():
#     from pydantic import BaseModel, Field
#     class PDFMetadataExtractedSchema(BaseModel):
#         title: str = Field(description="Title of the PDF file", required=True)
#         abstract: str = Field(description="Abstract of the PDF file", required=True)
#         keywords: List[str] = Field(description="List of keywords", required=True)
#         authors: List[str] = Field(description="List of authors", required=True)
#         year: int = Field(description="Year of the publication", required=True)
#     return PDFMetadataExtractedSchema


def get_structured_output_schema():
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "PDF Metadata Extraction Schema",
        "description": "Schema for PDF metadata extraction",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the academic paper. If the title is not in the content, then generate it from the abstract.",
                "minLength": 0,
            },
            "abstract": {
                "type": "string",
                "description": "The abstract or summary of the paper. If the abstract is not in the content, then generate it from the title.",
                "minLength": 1000,
            },
            "year": {
                "type": "integer",
                "description": "Publication year. If the year is not in the content, then return -1. If the year is not in the content, then return -1.",
                "minimum": -1,
                "maximum": 2100,
                "default": -1,
            },
            "authors": {
                "type": "array",
                "description": "List of paper authors. If the authors are not in the content, DO NOT GENERATE THEM, then return empty list.",
                "items": {"type": "string", "minLength": 0, "maxLength": 20},
                "minItems": 0,
                "uniqueItems": True,
            },
            "keywords": {
                "type": "array",
                "description": "List of keywords associated with the paper. If the keywords are not in the content, then generate them from the content.",
                "items": {"type": "string", "minLength": 0, "maxLength": 100},
                "minItems": 0,
                "uniqueItems": True,
            },
        },
        "required": ["title", "abstract", "year", "authors", "keywords"],
        "additionalProperties": False,
    }


class StructuredOutputExtractStrategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._config = {
            "configurable": {
                "thread_id": uuid.uuid4().hex,
            },
        }

    def __call__(self, state: PDFAnalyserState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")

        pdf_document = state.get("pdf_document")

        json_schema = get_structured_output_schema()
        messages = [
            SystemMessage(
                content=get_prompt_template_for_structured_output(
                    pdf_content=pdf_document.text.text,
                    think=False,
                )
            ),
        ]
        structured_llm = self._llm.with_structured_output(
            schema=json_schema,
            method="json_schema",
        )
        response = structured_llm.invoke(
            messages,
            config=self._config,
        )
        pdf_metadata_extracted = PDFMetadataExtracted(
            title=response.get("title", ""),
            abstract=response.get("abstract", ""),
            authors=response.get("authors", []),
            keywords=response.get("keywords", []),
            year=response.get("year", 0),
        )
        output = {
            "pdf_metadata_extracted": pdf_metadata_extracted,
        }
        lg.trace("end")
        return output


if __name__ == "__main__":
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        # model="qwen3:8b",
        model="qwen3:32b",
        temperature=0.0,
        num_predict=32000,
    )
    # from langchain.chat_models import ChatOpenAI
    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.0,
    # )
    pdf_document = PDFDocument(
        text=PDFText(
            text=(
                "DrugEx v2: de novo design of drug"
                "molecules by Pareto-based multi-objective "
                "reinforcement learning in polypharmacology"
                "Xuhan Liu1 , Kai Ye2 , Herman W. T. van Vlijmen1,3 , Michael T. M. Emmerich4 , Adriaan P. IJzerman1  and "
                "Gerard J. P. van Westen1*  "
                "Abstract In polypharmacology drugs are required to bind to multiple specific targets, for example to enhance efficacy or "
                "to reduce resistance formation. Although deep learning has achieved a breakthrough in de novo design in drug "
                "discovery, most of its applications only focus on a single drug target to generate drug-like active molecules. However, "
                "in reality drug molecules often interact with more than one target which can have desired (polypharmacology) or "
                "undesired (toxicity) effects. In a previous study we proposed a new method named DrugEx that integrates an explora-"
                "tion strategy into RNN-based reinforcement learning to improve the diversity of the generated molecules. Here, we "
                "extended our DrugEx algorithm with multi-objective optimization to generate drug-like molecules towards multiple "
                "targets or one specific target while avoiding off-targets (the two adenosine receptors,  A1AR and  A2AAR, and the potas-"
                "sium ion channel hERG in this study). In our model, we applied an RNN as the agent and machine learning predictors "
                "as the environment. Both the agent and the environment were pre-trained in advance and then interplayed under a "
                "reinforcement learning framework. The concept of evolutionary algorithms was merged into our method such that "
                "crossover and mutation operations were implemented by the same deep learning model as the agent. During the "
                "training loop, the agent generates a batch of SMILES-based molecules. Subsequently scores for all objectives provided "
                "by the environment are used to construct Pareto ranks of the generated molecules. For this ranking a non-dominated "
                "sorting algorithm and a Tanimoto-based crowding distance algorithm using chemical fingerprints are applied. Here, "
                "we adopted GPU acceleration to speed up the process of Pareto optimization. The final reward of each molecule is "
                "calculated based on the Pareto ranking with the ranking selection algorithm. The agent is trained under the guidance "
                "of the reward to make sure it can generate desired molecules after convergence of the training process. All in all we "
                "demonstrate generation of compounds with a diverse predicted selectivity profile towards multiple targets, offering "
                "the potential of high efficacy and low toxicity."
                "Keywords:  Deep learning, Adenosine receptors, Cheminformatics, Reinforcement learning, Multi-objective "
                "optimization, Exploration strategy"
            )
        ),
        metadata=PDFMetadata(
            title="Test",
            authors=["John Doe"],
            keywords=["test", "test2"],
            year=2021,
        ),
    )
    node = StructuredOutputExtractStrategy(llm=llm)
    output = node(state={"pdf_document": pdf_document})
    pprint(output)


# %%
class MetadataExtractorType(Enum):
    WITH_STRUCTURED_OUTPUT = lambda *a, **kw: StructuredOutputExtractStrategy(*a, **kw)  # noqa: E731
    RESPONSE_SCHEMAS = lambda *a, **kw: ResponseSchemasMetadataExtractStrategy(*a, **kw)  # noqa: E731


class PDFMetadataExtractorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._strategy = MetadataExtractorType.WITH_STRUCTURED_OUTPUT(llm=llm)

    def set_strategy(self, strategy: IStrategy):
        self._strategy = strategy

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        try:
            output = self._strategy(state)
            lg.success("Successfully extracted metadata")
        except Exception as e:
            lg.error(f"Failed to extract metadata: {e}")
            raise e
        lg.trace("end")
        return output


if __name__ == "__main__":
    import os

    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        # model="qwen3:8b",
        model="qwen3:32b",
        temperature=0.0,
        num_predict=32000,
    )
    # from langchain.chat_models import ChatOpenAI
    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.0,
    # )

    pdf_document = PDFDocument(
        text=PDFText(
            text=(
                "DrugEx v2: de novo design of drug"
                "molecules by Pareto-based multi-objective "
                "reinforcement learning in polypharmacology"
                "Xuhan Liu1 , Kai Ye2 , Herman W. T. van Vlijmen1,3 , Michael T. M. Emmerich4 , Adriaan P. IJzerman1  and "
                "Gerard J. P. van Westen1*  "
                "Abstract In polypharmacology drugs are required to bind to multiple specific targets, for example to enhance efficacy or "
                "to reduce resistance formation. Although deep learning has achieved a breakthrough in de novo design in drug "
                "discovery, most of its applications only focus on a single drug target to generate drug-like active molecules. However, "
                "in reality drug molecules often interact with more than one target which can have desired (polypharmacology) or "
                "undesired (toxicity) effects. In a previous study we proposed a new method named DrugEx that integrates an explora-"
                "tion strategy into RNN-based reinforcement learning to improve the diversity of the generated molecules. Here, we "
                "extended our DrugEx algorithm with multi-objective optimization to generate drug-like molecules towards multiple "
                "targets or one specific target while avoiding off-targets (the two adenosine receptors,  A1AR and  A2AAR, and the potas-"
                "sium ion channel hERG in this study). In our model, we applied an RNN as the agent and machine learning predictors "
                "as the environment. Both the agent and the environment were pre-trained in advance and then interplayed under a "
                "reinforcement learning framework. The concept of evolutionary algorithms was merged into our method such that "
                "crossover and mutation operations were implemented by the same deep learning model as the agent. During the "
                "training loop, the agent generates a batch of SMILES-based molecules. Subsequently scores for all objectives provided "
                "by the environment are used to construct Pareto ranks of the generated molecules. For this ranking a non-dominated "
                "sorting algorithm and a Tanimoto-based crowding distance algorithm using chemical fingerprints are applied. Here, "
                "we adopted GPU acceleration to speed up the process of Pareto optimization. The final reward of each molecule is "
                "calculated based on the Pareto ranking with the ranking selection algorithm. The agent is trained under the guidance "
                "of the reward to make sure it can generate desired molecules after convergence of the training process. All in all we "
                "demonstrate generation of compounds with a diverse predicted selectivity profile towards multiple targets, offering "
                "the potential of high efficacy and low toxicity."
                "Keywords:  Deep learning, Adenosine receptors, Cheminformatics, Reinforcement learning, Multi-objective "
                "optimization, Exploration strategy"
            )
        ),
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
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        pdf_document = state["pdf_document"]
        short_long_pdf_length_threshold = state["short_long_pdf_length_threshold"]
        lg.debug(f"Length of pdf_document: {len(pdf_document)}")
        lg.debug(f"Short_long_threshold: {short_long_pdf_length_threshold}")
        if len(pdf_document) > short_long_pdf_length_threshold:
            lg.trace("long_pdf")
            return PDFLength.LONG
        else:
            lg.trace("short_pdf")
            return PDFLength.SHORT


# %%
class PDFTextSplitterNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
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
        lg.trace("end")
        return output


# %%
class ResponseSchemasChunksMetadataExtractStrategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        pdf_chunks = state["pdf_chunks"]
        pdf_chunks_metadata_extracted = []

        response_schemas = get_response_schemas()
        prompt_template = get_prompt_template(
            think=True,
            pdf_content="{pdf_content}",
            format_instructions="{format_instructions}",
        )
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["pdf_content"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | self._llm | output_parser

        for pdf_chunk in pdf_chunks:
            response = chain.invoke(
                {
                    "pdf_content": pdf_chunk.text,
                }
            )
            pdf_chunk_metadata_extracted = PDFMetadataExtracted(
                title=response.get("title", ""),
                abstract=response.get("abstract", ""),
                keywords=response.get("keywords", []),
                authors=response.get("authors", []),
                year=response.get("year", 0),
            )
            pdf_chunks_metadata_extracted.append(pdf_chunk_metadata_extracted)

        output = {
            "pdf_chunks_metadata_extracted": pdf_chunks_metadata_extracted,
        }
        lg.trace("end")
        return output


# %%
class StructuredOutputChunksMetadataExtractStrategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._config = {
            "configurable": {
                "thread_id": uuid.uuid4().hex,
            },
        }

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        pdf_chunks = state["pdf_chunks"]

        think = False
        system_prompt = get_prompt_template_for_structured_output(
            think=think,
        )
        json_schema = get_structured_output_schema()
        structured_llm = self._llm.with_structured_output(
            schema=json_schema,
            method="json_schema",
        )
        pdf_chunks_metadata_extracted = []
        for pdf_chunk in pdf_chunks:
            system_prompt += f"PDF CONTENT:\n{pdf_chunk.text}\n\n"
            messages = [
                SystemMessage(content=system_prompt),
            ]
            response = structured_llm.invoke(
                messages,
                config=self._config,
            )
            lg.debug(f"Chunk {pdf_chunk.chunk_number}")
            lg.debug(f"Response: {response}")
            pdf_chunk_metadata_extracted = PDFMetadataExtracted(
                title=response.get("title", ""),
                abstract=response.get("abstract", ""),
                keywords=response.get("keywords", []),
                authors=response.get("authors", []),
                year=response.get("year", -1),
            )
            pdf_chunks_metadata_extracted.append(pdf_chunk_metadata_extracted)
        output = {
            "pdf_chunks_metadata_extracted": pdf_chunks_metadata_extracted,
        }
        lg.trace("end")
        return output


if __name__ == "__main__":
    import os

    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:32b",
        temperature=0.0,
        num_predict=32000,
    )
    # from langchain.chat_models.gigachat import GigaChat
    # # auth = os.getenv("GIGACHAT_CREDENTIALS")
    # llm = GigaChat(
    #     credentials="ZThkYmEyZWQtNjdjMi00NjBmLWE4MmUtMTM0NzRjNTZmMDM2OjY2YTMxMzA1LTlhNTQtNDNhOS05ZjIwLTgyMmMzOGIyODM5Nw==",
    #     model="GigaChat:lite",  # GigaChat:max, GigaChat:lite, GigaChat:pro
    #     scope="GIGACHAT_API_CORP",
    #     verify_ssl_certs=False,
    #     profanity_check=False,
    # )
    node = StructuredOutputChunksMetadataExtractStrategy(llm=llm)
    output = node(
        state={
            "pdf_chunks": [
                PDFContentChunk(
                    text="This chapter introduces a novel curiosity-driven unsupervised "
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
                    "sample efficiency and learning performance."
                    "Hongsheng Qian"
                    "Chao Sun"
                    "Chao Miao"
                    "Curiosity-Driven Learning"
                    "Offline Reinforcement Learning"
                    "Adaptive Temporal Distances"
                    "Data Collection"
                    "Self-Adaptive Hot Cognition"
                    "A Curiosity-Driven Unsupervised Data Collection Method with "
                    "Adaptive Temporal Distances for Offline Reinforcement Learning"
                    "2023",
                    chunk_number=0,
                ),
                PDFContentChunk(
                    text="This chapter introduces a contrastive-curiosity-driven learning "
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
                    chunk_number=1,
                ),
                PDFContentChunk(
                    text="The proposed input selection process based on curiosity can be "
                    "seen as a selective adversary to the representation learning "
                    "process by the contrastive loss in Eq. 3.1. This is because the "
                    "agent curiously selects samples that are challenging for "
                    "matching, which maximizes the encoder loss L 𝑓, while at the "
                    "same time it improves the encoder with difficult inputs by "
                    "minimizing L 𝑓. As a result, the agent can learn an improved "
                    "encoder that is more robust to different views of observations "
                    "with less training required, compared to using random inputs."
                    "Author(s) not specified in the provided content.",
                    chunk_number=2,
                ),
                PDFContentChunk(
                    text="The detailed setting of hyper-parameters is provided in Table A.3 and Table A.4."
                    "C. Sun"
                    "H. Qian"
                    "C. Miao"
                    "Contrastive Learning"
                    "Reinforcement Learning"
                    "Sample Efficiency"
                    "Curiosity-driven Exploration"
                    "MiniGrid"
                    "CCLF: Contrastive Curiosity-Driven Learning for Sample-Efficient Reinforcement Learning"
                    "2021",
                    chunk_number=3,
                ),
            ]
        },
    )
    pprint(output)


# %%
class ChunksMetadataExtractType(Enum):
    WITH_STRUCTURED_OUTPUT = lambda *a, **kw: StructuredOutputChunksMetadataExtractStrategy(*a, **kw)  # noqa: E731
    RESPONSE_SCHEMAS = lambda *a, **kw: ResponseSchemasChunksMetadataExtractStrategy(*a, **kw)  # noqa: E731


class ChunksMetadataExtractNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._strategy = ChunksMetadataExtractType.WITH_STRUCTURED_OUTPUT(llm=llm)

    def set_strategy(self, strategy: ChunksMetadataExtractType):
        self._strategy = strategy
        return self

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        try:
            output = self._strategy(state)
            lg.success("Successfully extracted metadata from chunks")
        except Exception as e:
            lg.error(f"Failed to extract metadata from chunks: {e}")
            raise e
        lg.trace("end")
        return output


if __name__ == "__main__":
    import os

    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:32b",
        temperature=0.0,
        num_predict=32000,
    )
    # from langchain.chat_models.gigachat import GigaChat
    # # auth = os.getenv("GIGACHAT_CREDENTIALS")
    # llm = GigaChat(
    #     credentials="ZThkYmEyZWQtNjdjMi00NjBmLWE4MmUtMTM0NzRjNTZmMDM2OjY2YTMxMzA1LTlhNTQtNDNhOS05ZjIwLTgyMmMzOGIyODM5Nw==",
    #     model="GigaChat:lite",  # GigaChat:max, GigaChat:lite, GigaChat:pro
    #     scope="GIGACHAT_API_CORP",
    #     verify_ssl_certs=False,
    #     profanity_check=False,
    # )
    node = ChunksMetadataExtractNode(llm=llm)
    output = node(
        state={
            "pdf_chunks": [
                PDFContentChunk(
                    text="This chapter introduces a novel curiosity-driven unsupervised "
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
                    "sample efficiency and learning performance."
                    "Hongsheng Qian"
                    "Chao Sun"
                    "Chao Miao"
                    "Curiosity-Driven Learning"
                    "Offline Reinforcement Learning"
                    "Adaptive Temporal Distances"
                    "Data Collection"
                    "Self-Adaptive Hot Cognition"
                    "A Curiosity-Driven Unsupervised Data Collection Method with "
                    "Adaptive Temporal Distances for Offline Reinforcement Learning"
                    "2023",
                    chunk_number=0,
                ),
                PDFContentChunk(
                    text="This chapter introduces a contrastive-curiosity-driven learning "
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
                    chunk_number=1,
                ),
                PDFContentChunk(
                    text="The proposed input selection process based on curiosity can be "
                    "seen as a selective adversary to the representation learning "
                    "process by the contrastive loss in Eq. 3.1. This is because the "
                    "agent curiously selects samples that are challenging for "
                    "matching, which maximizes the encoder loss L 𝑓, while at the "
                    "same time it improves the encoder with difficult inputs by "
                    "minimizing L 𝑓. As a result, the agent can learn an improved "
                    "encoder that is more robust to different views of observations "
                    "with less training required, compared to using random inputs."
                    "Author(s) not specified in the provided content.",
                    chunk_number=2,
                ),
                PDFContentChunk(
                    text="The detailed setting of hyper-parameters is provided in Table A.3 and Table A.4."
                    "C. Sun"
                    "H. Qian"
                    "C. Miao"
                    "Contrastive Learning"
                    "Reinforcement Learning"
                    "Sample Efficiency"
                    "Curiosity-driven Exploration"
                    "MiniGrid"
                    "CCLF: Contrastive Curiosity-Driven Learning for Sample-Efficient Reinforcement Learning"
                    "2021",
                    chunk_number=3,
                ),
            ]
        },
    )
    pprint(output)


# %%
def get_prompt_template_for_union(
    pdf_chunks_metadata_extracted: str | None = None,
    format_instructions: str | None = None,
    think: bool = False,
) -> str:
    prompt = (
        "You are a helpful assistant that helps to unite the metadata from the chunks of the analyzed parts of the PDF file. "
        "Look at all the metadata in the sequential chunks of one PDF file and make conclusions about which metadata is the best for the full description of the PDF file. "
        "\nABSTRACT reformulate it to be a single, more understandable, more informative, combining and voluminous."
        "\nKEYWORDS you can enumerate, but do not duplicate. Do not invent new keywords. The most frequent and important keywords should be in the beginning of the list."
        "\nAUTHORS try to take from the first chunk. Do not invent new authors."
        "\nYEAR try to take from the first chunk. Do not invent new years. Do not unite chunks."
        "\nTITLE try to take from the first chunk. Do not invent new titles. Do not unite chunks."
    )
    if not think:
        prompt += "\n/no-think\n\n"
    if format_instructions is not None:
        prompt += f"\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
    if pdf_chunks_metadata_extracted is not None:
        prompt += f"\nPDF CHUNKS METADATA EXTRACTED:\n{pdf_chunks_metadata_extracted}\n\n"
    return prompt


class ResponseSchemasChunksMetadataUnionStrategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        pdf_chunks_metadata_extracted = state["pdf_chunks_metadata_extracted"]
        if len(pdf_chunks_metadata_extracted) == 0:
            raise ValueError("pdf_chunks_metadata_extracted is empty")

        response_schemas = get_response_schemas()
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt_template = get_prompt_template_for_union(
            pdf_chunks_metadata_extracted="{pdf_chunks_metadata_extracted}",
            format_instructions="{format_instructions}",
            think=False,
        )
        lg.debug(f"prompt_template: {prompt_template}")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["pdf_chunks_metadata_extracted"],
            partial_variables={"format_instructions": format_instructions},
        )
        lg.debug(f"prompt: {prompt}")

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
        lg.trace("end")
        return output


class StructuredOutputChunksMetadataUnionStrategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._config = {"configurable": {"thread_id": uuid.uuid4().hex}}

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        pdf_chunks_metadata_extracted = state["pdf_chunks_metadata_extracted"]
        lg.debug(f"PDF chunks metadata extracted: {pdf_chunks_metadata_extracted}")

        json_schema = get_structured_output_schema()
        messages = [
            SystemMessage(
                content=get_prompt_template_for_union(
                    pdf_chunks_metadata_extracted=pdf_chunks_metadata_extracted,
                    think=False,
                )
            ),
        ]
        lg.debug(f"Messages: {messages}")

        structured_llm = self._llm.with_structured_output(
            schema=json_schema,
            method="json_schema",
        )
        response = structured_llm.invoke(
            messages,
            config=self._config,
        )
        lg.debug(f"Response: {response}")

        pdf_metadata_extracted = PDFMetadataExtracted(
            title=response.get("title", ""),
            abstract=response.get("abstract", ""),
            authors=response.get("authors", []),
            keywords=response.get("keywords", []),
            year=response.get("year", 0),
        )
        output = {
            "pdf_metadata_extracted": pdf_metadata_extracted,
        }
        lg.trace("end")
        return output


class PDFChunksMetadataUnionType(Enum):
    RESPONSE_SCHEMAS = lambda *a, **kw: ResponseSchemasChunksMetadataUnionStrategy(*a, **kw)  # noqa: E731
    WITH_STRUCTURED_OUTPUT = lambda *a, **kw: StructuredOutputChunksMetadataUnionStrategy(*a, **kw)  # noqa: E731


class PDFChunksMetadataUnionNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._strategy = PDFChunksMetadataUnionType.WITH_STRUCTURED_OUTPUT(llm=llm)

    def set_strategy(self, strategy: PDFChunksMetadataUnionType):
        self._strategy = strategy
        return self

    def __call__(self, state: PDFAnalyserState) -> PDFAnalyserState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        try:
            output = self._strategy(state)
            lg.success("Successfully united metadata from chunks")
        except Exception as e:
            lg.error(f"Failed to unite metadata from chunks: {e}")
            raise e
        lg.trace("end")
        return output


if __name__ == "__main__":
    # from langchain.chat_models import ChatOpenAI

    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.0,
    # )
    import os

    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:32b",
        temperature=0.0,
        num_predict=32000,
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
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        graph = StateGraph(PDFAnalyserState)
        graph.add_node("pdf_metadata_extract", PDFMetadataExtractorNode(llm=self._llm))
        graph.add_node("pdf_text_splitter", PDFTextSplitterNode(llm=self._llm))
        graph.add_node("pdf_chunks_metadata_extract", ChunksMetadataExtractNode(llm=self._llm))
        graph.add_node("pdf_chunks_metadata_union", PDFChunksMetadataUnionNode(llm=self._llm))

        graph.add_conditional_edges(
            START,
            PDFLengthConditionalEdge(),
            {
                PDFLength.SHORT: "pdf_metadata_extract",
                PDFLength.LONG: "pdf_text_splitter",
            },
        )
        graph.add_edge("pdf_metadata_extract", END)
        graph.add_edge("pdf_text_splitter", "pdf_chunks_metadata_extract")
        graph.add_edge("pdf_chunks_metadata_extract", "pdf_chunks_metadata_union")
        graph.add_edge("pdf_chunks_metadata_union", END)
        compiled_graph = graph.compile()
        lg.trace("end")
        return compiled_graph


if __name__ == "__main__":
    # llm = ChatOllama(
    #     model="hf.co/unsloth/Qwen3-30B-A3B-128K-GGUF:Q4_1",
    #     temperature=0.0,
    #     max_tokens=128000,
    # )

    # from langchain.chat_models import ChatOpenAI
    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.00,
    # )
    import os

    from langchain_ollama import ChatOllama

    from relepaper.domains.langgraph.workflows.utils.graph_displayer import (
        DisplayMethod,
        GraphDisplayer,
    )

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:32b",
        temperature=0.0,
        num_predict=128000,
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
            "short_long_pdf_length_threshold": 50000,
            "max_chunk_length": 50000,
            "max_chunks_count": 10,
            "intersection_length": 1000,
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
