# %%
import operator
import uuid
from pathlib import Path
from pprint import pprint
from typing import Annotated, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from relepaper.domains.langgraph.entities.relevance_decision import Threshold
from relepaper.domains.langgraph.entities.session import Session
from relepaper.domains.langgraph.interfaces import IWorkflowBuilder, IWorkflowNode
from relepaper.domains.langgraph.workflows.openalex_download_workflow import (
    OpenAlexDownloadState,
    OpenAlexDownloadWorkflowBuilder,
)
from relepaper.domains.langgraph.workflows.pdf_analyser import PDFAnalyserState
from relepaper.domains.langgraph.workflows.query_interpretator_workflow import (
    QueryInterpretatorState,
    QueryInterpretatorWorkflowBuilder,
)
from relepaper.domains.langgraph.workflows.relevance_evaluator import (
    RelevanceEvaluatorState,
    RelevanceEvaluatorWorkflowBuilder,
)
from relepaper.domains.langgraph.workflows.relevance_manager import (
    RelevanceManagerState,
    RelevanceManagerWorkflowBuilder,
)
from relepaper.domains.openalex.entities.pdf import PDFDownloadStrategy
from relepaper.domains.openalex.external.adapters.works_search.factory import (
    OpenAlexWorksSearchAdapterFactory,
)
from relepaper.domains.openalex.external.repositories.works.on_filesystem_repository import (
    OnFileSystemWorksRepository,
)
from relepaper.domains.openalex.services.download_service import OpenAlexPdfDownloadService
from relepaper.domains.openalex.services.works_save_service import OpenAlexWorksSaveService
from relepaper.domains.openalex.services.works_search_service import OpenAlexWorksSearchService


# %%
class GeneralWorkflowState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session: Session
    query_interpretator_state: QueryInterpretatorState
    openalex_download_state: OpenAlexDownloadState
    relevance_evaluator_state: RelevanceEvaluatorState
    relevance_manager_state: RelevanceManagerState


# %%
class QueryInterpretatorNode(IWorkflowNode):
    def __init__(
        self,
        llm: BaseChatModel,
        workflow_builder: QueryInterpretatorWorkflowBuilder = None,
        max_concurrency: int = 2,
        max_retries: int = 5,
    ):
        if workflow_builder is None:
            workflow_builder = QueryInterpretatorWorkflowBuilder(llm=llm)
        self._workflow = workflow_builder.build(checkpointer=InMemorySaver())
        displayer = GraphDisplayer(self._workflow).set_strategy(DisplayMethod.MERMAID)
        displayer.display()

        self._config = {
            "configurable": {
                "max_concurrency": max_concurrency,
                "max_retries": max_retries,
                "thread_id": uuid.uuid4().hex,
            },
        }

    def __call__(self, state: GeneralWorkflowState) -> dict:
        logger.trace("QueryInterpretatorNode: __call__: start")
        state_input = state["query_interpretator_state"]
        state_input["session"] = state["session"]
        state_input["user_query"] = state["messages"][-1]
        state_output = self._workflow.invoke(
            input=state_input,
            config=self._config,
        )
        output = {
            "query_interpretator_state": state_output,
        }
        logger.trace("QueryInterpretatorNode: __call__: end")
        return output


# %%
class OpenAlexDownloadNode(IWorkflowNode):
    def __init__(
        self,
        workflow_builder: OpenAlexDownloadWorkflowBuilder = None,
        works_save_service: OpenAlexWorksSaveService = None,
        works_search_service: OpenAlexWorksSearchService = None,
        download_pdfs_service: OpenAlexPdfDownloadService = None,
        max_concurrency: int = 2,
        max_retries: int = 5,
    ):
        if works_save_service is None:
            works_save_service = OpenAlexWorksSaveService(
                repository=OnFileSystemWorksRepository(Path().cwd() / ".data" / "openalex_works"),
            )
        if works_search_service is None:
            works_search_service = OpenAlexWorksSearchService(
                adapter=OpenAlexWorksSearchAdapterFactory.create(
                    strategy="pyalex",
                    config={
                        "email": "mail@example.com",
                        "max_retries": 3,
                        "retry_backoff_factor": 0.5,
                    },
                ),
            )
        if download_pdfs_service is None:
            download_pdfs_service = OpenAlexPdfDownloadService(
                strategy=PDFDownloadStrategy("selenium"),
                dirname=Path().cwd() / ".data" / "openalex_pdfs",
            )
        if workflow_builder is None:
            workflow_builder = OpenAlexDownloadWorkflowBuilder(
                works_save_service=works_save_service,
                works_search_service=works_search_service,
                download_pdfs_service=download_pdfs_service,
                max_concurrency=max_concurrency,
            )
        self._workflow = workflow_builder.build(checkpointer=InMemorySaver())
        self._config = {
            "configurable": {
                "max_concurrency": max_concurrency,
                "max_retries": 5,
                "thread_id": uuid.uuid4().hex,
            },
        }
        displayer = GraphDisplayer(self._workflow).set_strategy(DisplayMethod.MERMAID)
        displayer.display()

    def __call__(self, state: GeneralWorkflowState) -> dict:
        logger.trace("OpenAlexDownloadNode: __call__: start")
        state_input = state["openalex_download_state"]
        state_input["session"] = state["session"]
        state_input["reformulated_queries"] = state["query_interpretator_state"]["reformulated_queries"]
        state_output = self._workflow.invoke(
            input=state_input,
            config=self._config,
        )
        output = {
            "openalex_download_state": state_output,
        }
        logger.trace("OpenAlexDownloadNode: __call__: end")
        return output


# %%
class RelevanceEvaluatorNode(IWorkflowNode):
    def __init__(
        self,
        llm: BaseChatModel,
        workflow_builder: RelevanceEvaluatorWorkflowBuilder = None,
        max_concurrency: int = 2,
        max_retries: int = 5,
    ):
        if workflow_builder is None:
            workflow_builder = RelevanceEvaluatorWorkflowBuilder(llm=llm)
        self._workflow = workflow_builder.build(checkpointer=InMemorySaver())
        self._config = {
            "configurable": {
                "max_concurrency": max_concurrency,
                "max_retries": max_retries,
                "thread_id": uuid.uuid4().hex,
            },
        }
        displayer = GraphDisplayer(self._workflow).set_strategy(DisplayMethod.MERMAID)
        displayer.display()

    def __call__(self, state: GeneralWorkflowState) -> GeneralWorkflowState:
        logger.trace("RelevanceEvaluatorNode: __call__: start")
        state_input = state["relevance_evaluator_state"]
        state_input["session"] = state["session"]
        state_input["user_query"] = state["query_interpretator_state"]["user_query"]
        state_input["works"] = state["openalex_download_state"]["works"]
        state_input["pdfs"] = state["openalex_download_state"]["pdfs"]

        state_output = self._workflow.invoke(input=state_input)

        for pdf, extracted_metadata, score in zip(
            state_output["pdfs"],
            state_output["pdfs_metadata_extracted"],
            state_output["relevance_scores"],
        ):
            logger.debug(f"User query: {state_input['user_query']}")
            logger.debug(f"Title: {getattr(extracted_metadata, 'title', '')}")
            logger.debug(f"PDF: {pdf.filename}\n\tscore: {score}")

        output = {"relevance_evaluator_state": state_output}
        logger.trace("RelevanceEvaluatorNode: __call__: end")
        return output


# %%
class RelevanceManagerNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel, max_concurrency: int = 2, max_retries: int = 5):
        self._workflow = RelevanceManagerWorkflowBuilder(llm=llm).build(checkpointer=InMemorySaver())
        self._config = {
            "configurable": {
                "max_concurrency": max_concurrency,
                "max_retries": max_retries,
                "thread_id": uuid.uuid4().hex,
            },
        }
        displayer = GraphDisplayer(self._workflow).set_strategy(DisplayMethod.MERMAID)
        displayer.display()

    def __call__(self, state: GeneralWorkflowState) -> GeneralWorkflowState:
        logger.trace(f"{self.__class__.__name__}: __call__: start")
        state_input = state["relevance_manager_state"]
        state_input["session"] = state["session"]
        state_input["relevance_scores"] = state["relevance_evaluator_state"]["relevance_scores"]
        state_output = self._workflow.invoke(
            input=state_input,
            config=self._config,
        )
        output = {
            "relevance_manager_state": state_output,
        }
        logger.trace(f"{self.__class__.__name__}: __call__: end")
        return output


# %%
class GeneralWorkflowBuilder(IWorkflowBuilder):
    def __init__(
        self,
        llm: BaseChatModel,
    ):
        self._llm = llm
        self._query_interpretator_node = QueryInterpretatorNode(llm=llm)
        self._openalex_download_node = OpenAlexDownloadNode()
        self._relevance_evaluator_node = RelevanceEvaluatorNode(llm=llm)
        self._relevance_manager_node = RelevanceManagerNode(llm=llm)

    def build(self, **kwargs) -> StateGraph:
        logger.trace("GeneralWorkflowBuilder: build: start")
        graph_builder = StateGraph(GeneralWorkflowState)
        graph_builder.add_node("QueryInterpretator", self._query_interpretator_node)
        graph_builder.add_node("OpenAlexDownloader", self._openalex_download_node)
        graph_builder.add_node("RelevanceEvaluator", self._relevance_evaluator_node)
        graph_builder.add_node("RelevanceManager", self._relevance_manager_node)
        graph_builder.add_edge(START, "QueryInterpretator")
        graph_builder.add_edge("QueryInterpretator", "OpenAlexDownloader")
        graph_builder.add_edge("OpenAlexDownloader", "RelevanceEvaluator")
        graph_builder.add_edge("RelevanceEvaluator", "RelevanceManager")
        graph_builder.add_edge("RelevanceManager", END)
        graph = graph_builder.compile(**kwargs)
        logger.trace("GeneralWorkflowBuilder: build: end")
        return graph


# %%
if __name__ == "__main__":
    from relepaper.config.logger import setup_logger
    from relepaper.domains.langgraph.workflows.utils.graph_displayer import (
        DisplayMethod,
        GraphDisplayer,
    )

    setup_logger(stream_level="TRACE")
    # import os
    # from langchain_ollama import ChatOllama

    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # llm = ChatOllama(
    #     # model="qwen3:8b",
    #     model="qwen3:32b",
    #     temperature=0.0,
    #     max_tokens=10000,
    # )
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.00,
    )
    user_query = HumanMessage(
        content=(
            "Я пишу диссертацию по теме: Обучение с подкреплением. Обучение в офлайн-режиме. "
            "Скачай все статьи по этой теме. "
            "/no-think"
        ),
    )
    session = Session()

    # ----- General workflow -----
    general_workflow = GeneralWorkflowBuilder(llm=llm).build(checkpointer=InMemorySaver())
    displayer = GraphDisplayer(general_workflow).set_strategy(DisplayMethod.MERMAID)
    displayer.display()

    state_start = GeneralWorkflowState(
        messages=[user_query],
        session=session,
        query_interpretator_state=QueryInterpretatorState(
            session=None,
            user_query=None,
            main_topic="",
            context_for_queries="",
            reformulated_queries_quantity=10,
            reformulated_queries=[],
            comment=None,
        ),
        openalex_download_state=OpenAlexDownloadState(
            session=None,
            reformulated_queries=[],
            works=[],
            pdfs=[],
            per_page_works=5,
            timeout=60,
        ),
        relevance_evaluator_state=RelevanceEvaluatorState(
            session=None,
            user_query=None,
            works=[],
            pdfs=[],
            pdfs_metadata_extracted=[],
            relevance_scores=[],
            pdf_analyser_state=PDFAnalyserState(
                pdf_document=None,
                pdf_metadata_extracted=None,
                pdf_chunks=[],
                pdf_chunks_metadata_extracted=[],
                short_long_pdf_length_threshold=135000,
                max_chunk_length=135000,
                max_chunks_count=10,
                intersection_length=1000,
            ),
        ),
        relevance_manager_state=RelevanceManagerState(
            session=None,
            relevance_scores=[],
            mean_score_overall_pdfs=None,
            decision_threshold=Threshold(value=50),
            relevance_decision=None,
        ),
    )
    config = {
        "configurable": {
            "max_concurrency": 4,
            "max_retries": 5,
            "thread_id": uuid.uuid4().hex,
        },
    }
    state_end = general_workflow.invoke(
        input=state_start,
        config=config,
    )
    pprint(state_end["query_interpretator_state"]["reformulated_queries"])
    pprint(state_end["openalex_download_state"]["pdfs"])
    print(f"relevance_scores: \n\t{state_end['relevance_evaluator_state']['relevance_scores']}")
    for container in state_end["relevance_evaluator_state"]["relevance_scores"]:
        print(f"mean: {container.mean}")
    print(f"mean_score_overall_pdfs: {state_end['relevance_manager_state']['mean_score_overall_pdfs']}")
    print(f"decision: \n\t{state_end['relevance_manager_state']['relevance_decision']}")

    # state_history = [sh for sh in general_workflow.get_state_history(config)]
    # pprint(state_history)

# %%
