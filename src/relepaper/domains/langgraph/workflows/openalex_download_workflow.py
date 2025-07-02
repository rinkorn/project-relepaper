# %%
import uuid
from datetime import datetime
from pprint import pprint
from typing import List, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from relepaper.domains.langgraph.entities.session import Session
from relepaper.domains.langgraph.interfaces import IWorkflowBuilder, IWorkflowNode
from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.openalex.entities.work import OpenAlexWork
from relepaper.domains.openalex.external.adapters.works_search.factory import OpenAlexWorksSearchAdapterFactory
from relepaper.domains.openalex.external.repositories.works.on_filesystem_repository import OnFileSystemWorksRepository
from relepaper.domains.openalex.services.download_service import OpenAlexPdfDownloadService
from relepaper.domains.openalex.services.works_save_service import OpenAlexWorksSaveService
from relepaper.domains.openalex.services.works_search_service import OpenAlexWorksSearchService


# %%
class OpenAlexDownloadState(TypedDict):
    session: Session  # Session
    reformulated_queries: List[str]  # List of reformulated queries
    works: List[OpenAlexWork]  # List of works
    pdfs: List[OpenAlexPDF]  # List of downloaded pdfs
    per_page_works: int  # Number of works to search per query
    timeout: int  # Timeout for the search


class OpenalexSearchNode(IWorkflowNode):
    def __init__(self, works_search_service: OpenAlexWorksSearchService, max_concurrency: int):
        self._works_search_service = works_search_service
        self._max_concurrency = max_concurrency

    def __call__(self, state: OpenAlexDownloadState) -> OpenAlexDownloadState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        reformulated_queries = state["reformulated_queries"]
        per_page_works = state["per_page_works"]

        all_works = []

        # Поиск с сохранением связи запрос -> works
        for query_index, query in enumerate(reformulated_queries):
            lg.info(f"Searching for query {query_index}: {query}")

            # query_uuid = str(uuid.uuid4().hex)
            query_works = self._works_search_service.search_works(
                query=query,
                per_page_works=per_page_works,
            )

            # Добавляем метаданные запроса к каждому work
            current_time = datetime.now()
            for work in query_works:
                work.source_query = query
                work.source_query_index = query_index
                work.found_at = current_time

            lg.info(f"Found {len(query_works)} works for query {query_index}")
            all_works.extend(query_works)

        lg.info(f"Total works found: {len(all_works)}")
        output = {
            "works": all_works,
        }
        lg.trace("end")
        return output


class DownloadWorksNode(IWorkflowNode):
    def __init__(self, works_save_service: OpenAlexWorksSaveService):
        self._works_save_service = works_save_service

    def __call__(self, state: OpenAlexDownloadState) -> OpenAlexDownloadState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        works = state["works"]
        self._works_save_service.save_works(works)
        output = {}
        lg.trace("end")
        return output


class DownloadPDFsNode(IWorkflowNode):
    def __init__(self, download_pdfs_service: OpenAlexPdfDownloadService):
        self._download_pdfs_service = download_pdfs_service

    def __call__(self, state: OpenAlexDownloadState) -> OpenAlexDownloadState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        works = state["works"]
        timeout = state["timeout"]

        works_with_pdf = [w for w in works if w.pdf_url]
        lg.info(f"Works with url-pdfs for downloading: {len(works_with_pdf)}")
        pdfs = self._download_pdfs_service.download_from_works(works_with_pdf, timeout=timeout)

        # Связываем PDF с исходными запросами через work
        for pdf, work in zip(pdfs, works_with_pdf):
            pdf.source_query = work.source_query
            pdf.source_work_id = work.id
            pdf.source_query_index = work.source_query_index

        lg.info(f"Downloaded pdfs: {len(pdfs)}")
        output = {
            "pdfs": pdfs,
        }
        lg.trace("end")
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
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
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
        lg.trace("end")
        return graph


if __name__ == "__main__":
    from relepaper.config.dev_settings import get_dev_settings
    from relepaper.config.logger import setup_logger
    from relepaper.domains.langgraph.workflows.utils.graph_displayer import (
        DisplayMethod,
        GraphDisplayer,
    )

    setup_logger(stream_level="INFO")

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
        strategy=PDFDownloadStrategy.SELENIUM,
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
    displayer = GraphDisplayer(workflow).set_strategy(DisplayMethod.MERMAID)
    displayer.display()

    session = Session()

    reformulated_queries = [
        "fluorescent mobility experiments with significant equipment",
        "research on fluorescent mobility using advanced equipment",
        "studies on fluorescent mobility with high-impact tools",
        # "investigations into fluorescent mobility through advanced instrumentation",
        # "experiments on fluorescent mobility utilizing state-of-the-art equipment",
    ]
    state_input = OpenAlexDownloadState(
        session=session,
        reformulated_queries=reformulated_queries,
        works=[],
        pdfs=[],
        per_page_works=2,
        timeout=30,
    )
    config = {
        "configurable": {
            "max_concurrency": 4,
            "max_retries": 5,
            "thread_id": uuid.uuid4().hex,
        },
    }
    state_output = workflow.invoke(
        input=state_input,
        config=config,
    )
    print(state_output.keys())

    state_history = [sh for sh in workflow.get_state_history(config)]
    pprint(state_history)

    # Получаем все works и pdfs
    works = state_output["works"]
    pdfs = state_output["pdfs"]

    # Группируем по запросам
    works_by_query = {}
    pdfs_by_query = {}

    for work in works:
        query = work.source_query
        if query not in works_by_query:
            works_by_query[query] = []
        works_by_query[query].append(work)

    for pdf in pdfs:
        query = pdf.source_query
        if query not in pdfs_by_query:
            pdfs_by_query[query] = []
        pdfs_by_query[query].append(pdf)

    # Выводим результаты по запросам
    for query, query_works in works_by_query.items():
        query_pdfs = pdfs_by_query.get(query, [])
        print(f"Запрос: {query}")
        print(f"  - Works: {len(query_works)}")
        print(f"  - PDFs: {len(query_pdfs)}")
        for work in query_works:
            print(f"    * {work.title}")
