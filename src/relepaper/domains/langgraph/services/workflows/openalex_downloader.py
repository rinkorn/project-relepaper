# %%
import logging
import operator
import uuid
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Annotated, List, Sequence, TypedDict

from IPython.display import Image, display
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import Tool
from langchain_core.tools import tool as tool_decorator
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from pydantic import Field

from relepaper.domains.openalex.entities.openalex_pdf import OpenAlexPDF
from relepaper.domains.openalex.services.pdf_download.factory import PDFDownloadServiceFactory

logger = logging.getLogger(__name__)


# %%
llm = ChatOllama(
    # model="qwen3:8b",
    # model="qwen3:14b",
    # model="qwen3:30B-a3b",
    # model="qwen3:32b",
    model="qwen2.5-coder:14b",
    temperature=0.0,
    max_tokens=10000,
)


# %%
def reformulate_query(llm: BaseChatModel, query: str, quantity: int = 10) -> str:
    """Reformulate the user query to another variants for more relevant articles search.

    Args:
        query: user query
        quantity: number of reformulated queries
    Returns:
        list of reformulated queries
    """

    messages = [
        SystemMessage(
            content=(
                f"Refomatulate the following user query to {quantity} variants for more relevant articles search."
                " The queries should cover different topics and not repeat."
                " They should be written in English."
                f" The answer should be a list of {quantity} reformulated queries without numbering,"
                " without repetitions, without additional comments and characters."
                "\n/no_think"
            )
        ),
        HumanMessage(
            content=(f"{query}"),
        ),
    ]
    json_schema = {
        "title": "query_list",
        "description": "list of queries",
        "type": "array",
        "items": {"type": "string"},
        "minItems": quantity,
        "maxItems": quantity,
    }
    structured_llm = llm.with_structured_output(
        schema=json_schema,
        method="json_schema",
    )
    queries = structured_llm.invoke(messages)
    return queries


if __name__ == "__main__":
    # tool_reformulate_query = Tool(
    #     name="reformulate_query",
    #     func=reformulate_query,
    #     description="Reformulate the user query to 10 variants for more relevant articles search.",
    # )
    query = "Я пишу диссертацию по теме: флюаресцентная подвижность. Эксперименты на значимом оборудовании."
    response = tool_decorator(reformulate_query).invoke(
        input={
            "llm": llm,
            "query": query,
            "quantity": 5,
        },
    )
    reformulated_queries = response
    pprint(reformulated_queries)


# %%
def openalex_search(query: str, per_page: int = 5) -> List[str]:
    """Search for articles on the OpenAlex hub.

    Args:
        query: user query
        per_page: number of works to return
    Returns:
        list of pdf urls
    """

    from pyalex import Works, config

    config.email = "mail@example.com"
    config.max_retries = 3
    config.retry_backoff_factor = 0.5

    W = Works().search_filter(title_and_abstract=query)
    # W = W.filter(is_oa=True)
    W = W.filter(has_oa_accepted_or_published_version=True)
    # W = W.select(["primary_location", "id", "title", "keywords", "doi"])
    # S = Sources().filter(works_count=">1000").get()
    # W = W.filter(institutions={"country_code": "ru"})
    # W = W.filter(institutions={"country_code": "!ru"})
    # W = W.filter(publication_year=2023)
    # W = W.group_by("institutions.country_code")
    W = W.sort(cited_by_count="desc")
    works = W.get(per_page=per_page)

    # with open(Path("works.json"), "w") as f:
    #     json.dump(works, f)
    # with open(Path("works.json")) as f:
    #     works = [Work(w) for w in json.load(f)]

    return works


if __name__ == "__main__":
    query = "Я пишу диссертацию по теме: флюаресцентная подвижность. Эксперименты на значимом оборудовании."
    reformulated_queries = reformulate_query(llm, query)
    works = tool_decorator(openalex_search).batch(
        inputs=[{"query": q, "per_page": 5} for q in reformulated_queries],
        config={"max_concurrency": 4},
    )
    pprint(len([work for sublist in works for work in sublist]))


# %%
def extract_pdf_url(work: dict) -> List[str]:
    """Extract pdf urls from works.

    Args:
        works: list of works
    Returns:
        list of pdf urls
    """
    try:
        pdf_url = work.get("primary_location", {}).get("pdf_url", None)
    except Exception:
        logger.error(f"Error: {work}")
        return None
    return pdf_url


if __name__ == "__main__":
    pdf_urls = tool_decorator(extract_pdf_url).batch(
        inputs=[{"work": work} for work in [work for sublist in works for work in sublist]],
        config={"max_concurrency": 4},
    )
    pprint(pdf_urls)


# %%
def extract_work_id(work: dict) -> str:
    """Extract pdf urls from works.

    Args:
        works: list of works
    Returns:
        work id
    """
    web_id = work.get("id", None)
    if web_id:
        return web_id.split("/")[-1]
    else:
        return None


if __name__ == "__main__":
    works_ids = tool_decorator(extract_work_id).batch(
        inputs=[{"work": work} for work in [work for sublist in works for work in sublist]],
        config={"max_concurrency": 4},
    )
    pprint(works_ids)


# %%
def save_work(work: dict, dirname: Path) -> None:
    """Save works to a json file.

    Args:
        works: list of works
        path: path to save works
    """
    import json

    dirname.mkdir(parents=True, exist_ok=True)

    work_id = extract_work_id(work)
    work_path = dirname / f"{work_id}.json"
    with open(work_path, "w", encoding="utf-8") as f:
        json.dump(work, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    from relepaper.config.dev_settings import get_dev_settings

    session_id = "test_" + datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex
    dirname = get_dev_settings().project_path / "session" / session_id / "openalex_works"
    tool_decorator(save_work).batch(
        inputs=[
            {
                "work": work,
                "dirname": dirname,
            }
            for work in [work for sublist in works for work in sublist]
        ],
        config={"max_concurrency": 4},
    )


# %%
def load_work(dirname: Path, work_id: str) -> dict:
    """Load work from a json file.

    Args:
        dirname: directory name
        work_id: work id
    Returns:
        work
    """
    import json

    work_path = dirname / f"{work_id}.json"
    if not work_path.exists():
        return None
    with open(work_path, "r") as f:
        work = json.load(f)
    return work


if __name__ == "__main__":
    from relepaper.config.dev_settings import get_dev_settings

    dirname = get_dev_settings().project_path / "session" / session_id / "openalex_works"
    work_id = "W2047305537"
    work = tool_decorator(load_work).invoke(
        input={
            "dirname": dirname,
            "work_id": work_id,
        },
    )
    pprint(work)


# %%
def openalex_download_pdf(pdf_url: str, dirname: Path, strategy="selenium", timeout=10):
    """Download a PDF from the internet.

    Args:
        pdf_url: pdf url
        dirname: directory name
        strategy: strategy to download pdf ('requests', 'selenium', 'selenium_stealth')
        timeout: timeout for download
    Returns:
        OpenAlexPDF object
    """
    openalex_pdf = OpenAlexPDF(
        url=pdf_url,
        dirname=dirname,
    )
    try:
        service = PDFDownloadServiceFactory.create(strategy=strategy)
        service.download(openalex_pdf, timeout=timeout)
    except Exception as e:
        logger.error(f"Error: {e}")
    return openalex_pdf


if __name__ == "__main__":
    from relepaper.config.dev_settings import get_dev_settings

    pdf_urls = [url for url in pdf_urls if url]
    session_id = "test_" + datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex
    dirname = get_dev_settings().project_path / "session" / session_id / "openalex_pdfs"
    strategy = "selenium"
    timeout = 30
    pyalex_pdfs = tool_decorator(openalex_download_pdf).batch(
        inputs=[
            {
                "pdf_url": pdf_url,
                "dirname": dirname,
                "strategy": strategy,
                "timeout": timeout,
            }
            for pdf_url in pdf_urls[:10]
        ],
        config={"max_concurrency": 4},
    )
    pprint(pyalex_pdfs)


# %%
tools = {}
tools["reformulate_query"] = tool_decorator(reformulate_query)
tools["openalex_search"] = tool_decorator(openalex_search)
tools["openalex_download_pdf"] = tool_decorator(openalex_download_pdf)
tools["extract_pdf_url"] = tool_decorator(extract_pdf_url)
tools["extract_work_id"] = tool_decorator(extract_work_id)
tools["save_work"] = tool_decorator(save_work)
tools["load_work"] = tool_decorator(load_work)
# llm_with_tools = llm.bind_tools([t for t in tools.values()], tool_choice="any")


# %%
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str = Field(description="Session id")
    llm: BaseChatModel = Field(description="LLMChatModel")
    original_query: str = Field(description="Original user query")
    reformulate_query_quantity: int = Field(description="Quantity of reformulated queries")
    reformulated_queries: List[str] = Field(description="List of reformulated queries")
    pdf_urls: List[str] = Field(description="List of pdf urls")
    works: List[dict] = Field(description="List of works")
    works_ids: List[str] = Field(description="List of works ids")
    download_strategy: str = Field(description="Download strategy")
    download_timeout: int = Field(description="Download timeout")
    downloaded_pdfs: List[OpenAlexPDF] = Field(description="List of downloaded pdfs")


def reformulate_query_node(state: State):
    pprint(":::NODE: reformulate_query:::")
    if not state.get("original_query"):
        original_query = state["messages"][-1].content
    else:
        original_query = state["original_query"]
    reformulated_queries = tools["reformulate_query"].invoke(
        input={
            "llm": state["llm"],
            "query": original_query,
            "quantity": state["reformulate_query_quantity"],
        },
    )
    # reformulated_queries = llm_with_tools.invoke(original_query)
    # reformulated_queries = llm.invoke(original_query)
    pprint(reformulated_queries)
    function_message = FunctionMessage(
        content=reformulated_queries,
        name="reformulate_query",
    )
    output = {
        "original_query": original_query,
        "reformulated_queries": reformulated_queries,
        "messages": [function_message],
    }
    return output


def openalex_search_node(state: State):
    pprint(":::NODE: openalex_search:::")
    reformulated_queries = state["reformulated_queries"]
    # llm_with_tools = llm.bind_tools([tools["openalex_search"]])
    # response = llm_with_tools.invoke(state["messages"])
    works = tools["openalex_search"].batch(reformulated_queries)
    works = [work for sublist in works for work in sublist]
    pprint(f"works: {len(works)}")
    output = {
        "works": works,
    }
    return output


def extract_pdf_urls_node(state: State):
    pprint(":::NODE: extract_pdf_urls:::")
    works = state["works"]
    pdf_urls = tools["extract_pdf_url"].batch(
        inputs=[{"work": work} for work in works],
    )
    # pdf_urls = [url for url in pdf_urls if url]
    pprint(f"pdf_urls: {len(pdf_urls)}")
    output = {
        "pdf_urls": pdf_urls,
    }
    return output


def extract_works_ids_node(state: State):
    pprint(":::NODE: extract_works_ids:::")
    works = state["works"]
    works_ids = tools["extract_work_id"].batch(
        inputs=[{"work": work} for work in works],
    )
    # works_ids = [id for sublist in works_ids for work_id in sublist]
    pprint(f"works_ids: {len(works_ids)}")
    output = {
        "works_ids": works_ids,
    }
    return output


def save_works_node(state: State):
    pprint(":::NODE: save_works:::")
    works = state["works"]
    tools["save_work"].batch(
        inputs=[
            {
                "work": work,
                "dirname": get_dev_settings().project_path / "session" / state["session_id"] / "openalex_works",
            }
            for work in works
        ],
    )
    return {}


def openalex_download_pdfs_node(state: State):
    pprint(":::NODE: download_pdfs:::")
    pdf_urls = state["pdf_urls"]
    pdf_urls = [url for url in pdf_urls if url]
    pprint(f"pdf_urls for downloading: {len(pdf_urls)}")
    downloaded_pdfs = tools["openalex_download_pdf"].batch(
        inputs=[
            {
                "pdf_url": pdf_url,
                "dirname": get_dev_settings().project_path / "session" / state["session_id"] / "openalex_pdfs",
                "strategy": state["download_strategy"],
                "timeout": state["download_timeout"],
            }
            for pdf_url in pdf_urls
        ],
        config={"max_concurrency": 1},
    )
    pprint(f"downloaded_pdfs: {len(downloaded_pdfs)}")
    output = {
        "downloaded_pdfs": downloaded_pdfs,
    }
    return output


def should_continue_edge(state: State):
    pprint(":::CONDITION: should_continue:::")
    if len(state["downloaded_pdfs"]) > 4:
        return "end"
    else:
        return "continue"


# def call_model_node(state: AgentState):
#     print(":::NODE: call_model:::")
#     messages = state["messages"]
#     response = llm_with_tools.invoke(messages)
#     output = {
#         "messages": [response],
#         "api_call_count": state["api_call_count"],
#     }
#     return output


# def call_tool_node(state: AgentState):
#     print(":::NODE: call_tool:::")
#     tool = state["messages"][-1].tool_calls[0]
#     tool_name = tool["name"].lower()
#     tool_args = tool["args"]
#     tool_id = tool["id"]
#     tool_output = tools[tool_name].invoke(tool_args)
#     state["api_call_count"] += 1
#     pprint(f"Tool output: {tool_output}")
#     pprint(f"API call count after this tool call {state['api_call_count']}")
#     tool_message = ToolMessage(
#         content=tool_output,
#         tool_call_id=tool_id,
#     )
#     output = {
#         "messages": [tool_message],
#         "api_call_count": state["api_call_count"],
#     }
#     return output


workflow = StateGraph(State)
workflow.set_entry_point("reformulate_query")

workflow.add_node("reformulate_query", reformulate_query_node)
workflow.add_node("openalex_search", openalex_search_node)
workflow.add_node("extract_pdf_urls", extract_pdf_urls_node)
workflow.add_node("extract_works_ids", extract_works_ids_node)
workflow.add_node("save_works", save_works_node)
workflow.add_node("download_pdfs", openalex_download_pdfs_node)

workflow.add_edge("reformulate_query", "openalex_search")
workflow.add_edge("openalex_search", "extract_pdf_urls")
workflow.add_edge("extract_pdf_urls", "extract_works_ids")
workflow.add_edge("extract_works_ids", "save_works")
workflow.add_edge("extract_pdf_urls", "download_pdfs")
workflow.add_edge("save_works", END)
workflow.add_edge("download_pdfs", END)

workflow.add_conditional_edges(
    "download_pdfs",
    should_continue_edge,
    {
        "continue": "reformulate_query",
        "end": END,
    },
)

memory = InMemorySaver()
workflow_compiled = workflow.compile(checkpointer=memory)


def display_graph():
    try:
        # display(Image(workflow_compiled.get_graph(xray=True).draw_mermaid_png()))
        display(
            Image(
                workflow_compiled.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )
    except Exception:
        pass


display_graph()

# SystemMessage(
#     content=(
#         "You are responsible for answering user questions. "
#         "You use tools for that, These tools are available: "
#         f"{', '.join(tools.keys())}.\n"
#         "Use tools until you get the answer to the question. "
#         "If you don't get the answer, try to call the tool again. "
#         "You should not give up until you get the answer to the question. "
#         "You can use tools multiple times if needed. "
#         "You should not think about the answer, you should use tools. "
#         "Reply only with the answer to the question. "
#         "/no-think"
#     )
# ),

if __name__ == "__main__":
    started_messages = [
        HumanMessage(
            content=(
                "Я пишу диссертацию по теме: Обучение с подкреплением. "
                "Обучение в офлайн-режиме. "
                "Скачай все статьи по этой теме. /no-think"
            ),
        ),
    ]
    # started_state = AgentState(messages=messages, api_call_count=0)
    started_state = {
        "session_id": datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex,
        "messages": started_messages,
        "llm": llm,
        "reformulate_query_quantity": 5,
        "download_strategy": "selenium_stealth",
        "download_timeout": 6,
        # "original_query": "",
        # "downloaded_pdfs": [],
        # "reformulated_queries": [],
        # "pdf_urls": [],
    }

    config = {
        "configurable": {
            "max_concurrency": 4,
            "max_retries": 5,
            "thread_id": uuid.uuid4(),
        },
    }
    state = workflow_compiled.invoke(input=started_state, config=config)
    # pprint(state)

# %%
state_history = [sh for sh in workflow_compiled.get_state_history(config)]
pprint(state_history)


# %%
class OpenAlexWorkflowService:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: dict[str, Tool],
        prompt: SystemMessage | str,
        graph: StateGraph,
        state: State,
        config: dict,
    ):
        self.llm = llm
        self.tools = tools
        self.prompt = prompt
        self.graph = graph
        self.state = state
        self.config = config

    def invoke(self, query: str):
        state = self.workflow.invoke(
            input=self.state,
            config=self.config,
        )
        pprint(state)
        return state

    def get_state_history(self):
        return self.workflow.get_state_history(self.config)


if __name__ == "__main__":
    prompt = SystemMessage(
        content=(
            "You are responsible for answering user questions. "
            "You use tools for that, These tools are available: "
            f"{', '.join(tools.keys())}.\n"
        )
    )
    agent = OpenAlexWorkflowService(
        llm=llm,
        tools=tools,
        prompt=prompt,
        workflow_graph=workflow,
        state=started_state,
        config=config,
    )
    state = agent.invoke(
        query="Я пишу диссертацию по теме: Обучение с подкреплением. "
        "Обучение в офлайн-режиме. "
        "Скачай все статьи по этой теме. /no-think"
    )
    pprint(state)
    state_history = agent.get_state_history()
    pprint(state_history)

# %%
