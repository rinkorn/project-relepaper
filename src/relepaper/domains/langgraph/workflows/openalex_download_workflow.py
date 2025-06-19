# %%
import logging
import uuid
from pprint import pprint
from typing import List, TypedDict

from langchain_core.tools import tool as tool_decorator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from relepaper.domains.langgraph.entities.session import Session
from relepaper.domains.langgraph.workflows.interfaces import IWorkflowBuilder, IWorkflowNode
from relepaper.domains.langgraph.workflows.utils import display_graph
from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.openalex.entities.work import OpenAlexWork
from relepaper.domains.openalex.external.adapters.works_search.factory import OpenAlexWorksSearchAdapterFactory
from relepaper.domains.openalex.external.repositories.works.on_filesystem_repository import OnFileSystemWorksRepository
from relepaper.domains.openalex.services.download_service import OpenAlexPdfDownloadService
from relepaper.domains.openalex.services.works_save_service import OpenAlexWorksSaveService
from relepaper.domains.openalex.services.works_search_service import OpenAlexWorksSearchService

# %%
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    stream_formatter = logging.Formatter("__log__: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


# %%
class OpenAlexDownloadState(TypedDict):
    session: Session  # Session
    reformulated_queries: List[str]  # List of reformulated queries
    per_page: int  # Number of works to search per query
    timeout: int  # Timeout for the search
    works: List[OpenAlexWork]  # List of works
    pdfs: List[OpenAlexPDF]  # List of downloaded pdfs


class OpenalexSearchNode(IWorkflowNode):
    def __init__(self, works_search_service: OpenAlexWorksSearchService, max_concurrency: int):
        self._works_search_service = works_search_service
        self._max_concurrency = max_concurrency

    def __call__(self, state: OpenAlexDownloadState) -> OpenAlexDownloadState:
        logger.info(":::NODE: openalex_search:::")
        reformulated_queries = state["reformulated_queries"]
        per_page = state["per_page"]
        timeout = state["timeout"]
        works = tool_decorator(self._works_search_service.search_works).batch(
            inputs=[
                {
                    "query": q,
                    "per_page": per_page,
                    "timeout": timeout,
                }
                for q in reformulated_queries
            ],
            config={"max_concurrency": self._max_concurrency},
        )
        works = [work for sublist in works for work in sublist]
        logger.info(f"works: {len(works)}")
        output = {
            "works": works,
        }
        return output


class DownloadWorksNode(IWorkflowNode):
    def __init__(self, works_save_service: OpenAlexWorksSaveService):
        self._works_save_service = works_save_service

    def __call__(self, state: OpenAlexDownloadState) -> OpenAlexDownloadState:
        logger.info(":::NODE: save_works:::")
        works = state["works"]
        self._works_save_service.save_works(works)
        output = {}
        return output


class DownloadPDFsNode(IWorkflowNode):
    def __init__(self, download_pdfs_service: OpenAlexPdfDownloadService):
        self._download_pdfs_service = download_pdfs_service

    def __call__(self, state: OpenAlexDownloadState) -> OpenAlexDownloadState:
        logger.info(":::NODE: download_pdfs:::")
        works = state["works"]
        timeout = state["timeout"]
        works = [w for w in works if w.pdf_url]
        logger.info(f"works for downloading pdfs: {len(works)}")
        pdfs = self._download_pdfs_service.download_from_works(works, timeout=timeout)
        logger.info(f"downloaded pdfs: {len(pdfs)}")
        output = {
            "pdfs": pdfs,
        }
        return output


class OpenAlexDownloadWorkflowBuilder(IWorkflowBuilder):
    def __init__(
        self,
        works_search_service: OpenAlexWorksSearchService,
        works_save_service: OpenAlexWorksSaveService,
        download_pdfs_service: OpenAlexPdfDownloadService,
        max_concurrency: int = 4,
        **kwargs,
    ):
        self._works_search_service = works_search_service
        self._works_save_service = works_save_service
        self._download_pdfs_service = download_pdfs_service
        self._max_concurrency = max_concurrency
        self._kwargs = kwargs

    def build(self, **kwargs) -> StateGraph:
        graph_builder = StateGraph(OpenAlexDownloadState)
        graph_builder.add_node("OpenalexSearch", OpenalexSearchNode(self._works_search_service, self._max_concurrency))
        graph_builder.add_node("DownloadWorks", DownloadWorksNode(self._works_save_service))
        graph_builder.add_node("DownloadPDFs", DownloadPDFsNode(self._download_pdfs_service))
        graph_builder.add_edge(START, "OpenalexSearch")
        graph_builder.add_edge("OpenalexSearch", "DownloadWorks")
        graph_builder.add_edge("OpenalexSearch", "DownloadPDFs")
        graph_builder.add_edge("DownloadWorks", END)
        graph_builder.add_edge("DownloadPDFs", END)
        graph = graph_builder.compile(**kwargs)
        return graph


if __name__ == "__main__":
    from relepaper.config.dev_settings import get_dev_settings

    session = Session()
    datetime_str = session.created_at.strftime("%Y%m%dT%H%M%S")
    works_path = get_dev_settings().project_path / "session" / f"test_{datetime_str}_{session.id}" / "openalex_works"
    pdfs_path = get_dev_settings().project_path / "session" / f"test_{datetime_str}_{session.id}" / "openalex_pdfs"

    works_save_service = OpenAlexWorksSaveService(
        repository=OnFileSystemWorksRepository(works_path),
    )
    works_search_service = OpenAlexWorksSearchService(
        adapter=OpenAlexWorksSearchAdapterFactory.create(
            strategy="pyalex",
            config={
                "email": "mail@example.com",
                "max_retries": 3,
                "retry_backoff_factor": 0.5,
            },
        )
    )
    download_pdfs_service = OpenAlexPdfDownloadService(
        strategy=PDFDownloadStrategy("selenium"),
        dirname=pdfs_path,
    )
    workflow = OpenAlexDownloadWorkflowBuilder(
        works_save_service=works_save_service,
        works_search_service=works_search_service,
        download_pdfs_service=download_pdfs_service,
        max_concurrency=4,
    ).build(
        checkpointer=InMemorySaver(),
    )
    display_graph(workflow)

    session = Session()

    reformulated_queries = [
        "fluorescent mobility experiments with significant equipment",
        "research on fluorescent mobility using advanced equipment",
        "studies on fluorescent mobility with high-impact tools",
        "investigations into fluorescent mobility through advanced instrumentation",
        "experiments on fluorescent mobility utilizing state-of-the-art equipment",
    ]
    started_state = OpenAlexDownloadState(
        session=session,
        reformulated_queries=reformulated_queries,
        works=[],
        pdfs=[],
        per_page=5,
        timeout=30,
    )
    config = {
        "configurable": {
            "max_concurrency": 4,
            "max_retries": 5,
            "thread_id": uuid.uuid4().hex,
        },
    }
    state = workflow.invoke(
        input=started_state,
        config=config,
    )
    print(state.keys())

    state_history = [sh for sh in workflow.get_state_history(config)]
    pprint(state_history)
