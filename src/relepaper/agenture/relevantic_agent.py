# Терзать агента до тех пор пока ответ не вернется

# %%
import logging
import operator
from pprint import pprint
from typing import Annotated, List, Sequence, TypedDict

from IPython.display import Image, display
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate

# from langchain.schema import HumanMessage
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import tool as tool_decorator
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from relepaper.search.openalex.load_pdf import (
    load_pdf_with_requests,
    load_pdf_with_selenium,
    load_pdf_with_selenium_stealth,
)

logger = logging.getLogger(__name__)


# %%
# llm = ChatOpenAI(
#     base_url="http://localhost:7007/v1",
#     api_key="not_needed",
#     temperature=0.0,
# )
llm = ChatOllama(
    model="qwen3:30B-a3b",
    # model="qwen3:32b",
    # model="qwen2.5-coder:14b",
    temperature=0.0,
)


# %%
tools = {}
tools["load_pdf_with_requests"] = tool_decorator(load_pdf_with_requests)
tools["load_pdf_with_selenium"] = tool_decorator(load_pdf_with_selenium)
tools["load_pdf_with_selenium_stealth"] = tool_decorator(load_pdf_with_selenium_stealth)


# %%
# def reformulate_query(query: str) -> str:
#     """Reformulate the user query to 10 variants for more relevant articles search

#     Args:
#         query: user query

#     Returns:
#         list of reformulated queries
#     """

#     messages = [
#         SystemMessage(
#             content=(
#                 "Refomatulate the following user query to 10 variants for more relevant articles search."
#                 " The queries should cover different topics and not repeat."
#                 " They should be written in English."
#                 " The answer should be a list of 10 reformulated queries without numbering,"
#                 " without repetitions, without additional comments and characters."
#                 "\n/no_think"
#             )
#         ),
#         HumanMessage(
#             content=(f"{query}"),
#         ),
#     ]
#     json_schema = {
#         "title": "query_list",
#         "description": "list of queries",
#         "type": "array",
#         "items": {"type": "string"},
#         "minItems": 10,
#         "maxItems": 10,
#     }
#     structured_llm = llm.with_structured_output(
#         schema=json_schema,
#         method="json_schema",
#     )
#     response = structured_llm.invoke(messages)

#     # class ResponseFormatter(BaseModel):
#     #     """Always use this tool to structure your response to the user."""
#     #     queries: List[str] = Field(..., description="list of queries")
#     # structured_llm = llm.with_structured_output(
#     #     ResponseFormatter,
#     #     method="json_schema",
#     # )
#     # response = structured_llm.invoke(messages)
#     # response = response.queries

#     return response


# tools["reformulate_query"] = Tool(
#     name="reformulate_query",
#     func=reformulate_query,
#     description="Reformulate the user query to 10 variants for more relevant articles search.",
# )
# query = "Я пишу диссертацию по теме: флюаресцентная подвижность. Эксперименты на значимом оборудовании."
# response = tools["reformulate_query"].invoke(query)
# reformulated_queries = response
# pprint(reformulated_queries)


# %%
def reformulate_query(query: str) -> List[str]:
    """Reformulate the user query to 10 variants for more relevant articles search

    Args:
        query: user query

    Returns:
        list of reformulated queries
    """
    response_schemas = [
        ResponseSchema(
            name="queries",
            description="list of queries",
            type="array",
            items={"type": "string"},
            min_items=10,
            max_items=10,
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    template = (
        "Refomatulate the following user query to 10 variants for more relevant articles search."
        "\nThe queries should cover different topics and not repeat."
        "\nThey should be written in English."
        "\nThe answer should be a list of 10 reformulated queries without numbering,"
        "\nwithout repetitions, without additional comments and characters."
        "\n\n{format_instructions}"
        "\n\n{query}"
        # "\n/no_think"
    )
    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )
    chain = prompt | llm | output_parser
    response = chain.invoke({"query": query})
    queries = response["queries"]
    return queries


tools["reformulate_query"] = tool_decorator(reformulate_query)
# tools["reformulate_query"] = Tool(
#     name="reformulate_query",
#     func=reformulate_query,
#     description="Reformulate the user query to 10 variants for more relevant articles search.",
# )
# query = "query: Флюаресцентная подвижность. Эксперименты на значимом оборудовании."
query = "Я пишу диссертацию на тему: 'Флюаресцентная подвижность. Эксперименты на значимом оборудовании.'"
response = tools["reformulate_query"].invoke(query)
reformulated_queries = response
pprint(reformulated_queries)


# %%
def pyalex_search(query: str) -> List[str]:
    """Search for articles on the internet.

    Args:
        query: user query

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
    W = W.select(["primary_location"])
    # S = Sources().filter(works_count=">1000").get()
    # W = W.filter(institutions={"country_code": "ru"})
    # W = W.filter(institutions={"country_code": "!ru"})
    # W = W.filter(publication_year=2023)
    # W = W.group_by("institutions.country_code")
    W = W.sort(cited_by_count="desc")
    works = W.get(per_page=5)

    # with open(Path("works.json"), "w") as f:
    #     json.dump(works, f)
    # with open(Path("works.json")) as f:
    #     works = [Work(w) for w in json.load(f)]

    pdf_urls = []
    for work in works:
        # print(work)
        try:
            pdf_url = work["primary_location"].get("pdf_url", None)
            if pdf_url:
                pdf_urls.append(pdf_url)
        except Exception:
            pprint(work)
            print(f"Error: {work}")

    response = AIMessage(
        content=pdf_urls,
    )
    return response


tools["pyalex_search"] = tool_decorator(pyalex_search)
# query = "use of modern equipment for studying fluorescence"
# response = tools["pyalex_search"].invoke(query)
# pprint(response.content)

for q in reformulated_queries:
    response = tools["pyalex_search"].invoke(q)
    pprint(response.content)


# %%
def cycle_download_pdfs(query: str):
    """Download all PDFs from the internet.

    Args:
        query: user query

    Returns:
        list of downloaded pdfs
    """
    reformulated_queries = tools["reformulate_query"].invoke(query)
    downloaded_pdfs = []
    for query in reformulated_queries.content:
        print(query)
        response = tools["pyalex_search"].invoke(query)
        print(response.content)
        pdf_urls = response.content
        for pdf_url in pdf_urls:
            response = tools["load_pdf_with_selenium_stealth"].invoke(pdf_url)
            if response:
                downloaded_pdfs.append(pdf_url)

    response = AIMessage(
        content=downloaded_pdfs,
    )
    return response


tools["cycle_download_pdfs"] = tool_decorator(cycle_download_pdfs)
query = "Я пишу диссертацию по теме: флюаресцентная подвижность. Эксперименты на значимом оборудовании."
response = tools["cycle_download_pdfs"].invoke(query)
pprint(response.content)


# %%
llm_with_tools = llm.bind_tools([t for t in tools.values()], tool_choice="any")


# %%
messages = []
messages.append(
    SystemMessage(
        content=(
            "You are responsible for answering user questions. "
            "You use tools for that, These tools are available: "
            f"{', '.join(tools.keys())}.\n"
            "Use tools until you get the relevant articles. "
            "Нужно скачать в цикле все статьи по запросу пользователя. "
            "If you don't get the answer, try to call the tool again. "
            "You should not give up until you get the answer to the question. "
            "You can use tools multiple times if needed. "
            "You should not think about the answer, you should use tools. "
            "Reply only with the answer to the question. "
            "/no-think"
        )
    )
)
messages.append(
    HumanMessage(
        content=(
            "Я пишу диссертацию по теме: флюаресцентная подвижность."
            " Эксперименты на значимом оборудовании."
            " Скачай все статьи по этой теме."
        )
    )
)

llm_output = llm_with_tools.invoke(messages)
messages.append(llm_output)

for tool_call in llm_output.tool_calls:
    tool_name = tool_call["name"].lower()
    tool_args = tool_call["args"]
    tool_id = tool_call["id"]
    tool_output = tools[tool_name].invoke(tool_args)
    tool_message = ToolMessage(
        content=tool_output,
        tool_call_id=tool_id,
    )
    messages.append(tool_message)

llm_output2 = llm_with_tools.invoke(messages)
messages.append(llm_output2)

for msg in messages:
    pprint(f"{msg.type}: {msg.content}")


# %%
class AgentInput(BaseModel):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    api_call_count: int = 0
    reformulated_queries: List[str] = []
    downloaded_pdfs: List[str] = []
    paper_notes: List[str] = []


class AgentOutput(BaseModel):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    api_call_count: int = 0
    reformulated_queries: List[str] = []
    downloaded_pdfs: List[str] = []
    paper_notes: List[str] = []


class AgentState(TypedDict, input=AgentInput, output=AgentOutput):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    api_call_count: int = 0
    reformulated_queries: List[str] = []
    downloaded_pdfs: List[str] = []
    paper_notes: List[str] = []


def route_tools_node(state: AgentState):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def shoud_continue_edge(state: AgentState):
    print(":::CONDITION: shoud_continue:::")
    messages = state["messages"]
    last_messages = messages[-1]
    if not last_messages.tool_calls:
        pprint(last_messages.tool_calls)
        pprint(last_messages.content)
        return "end"
    else:
        return "continue"


def call_model_node(state: AgentState):
    print(":::NODE: call_model:::")
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    output = {
        "messages": [response],
        "api_call_count": state["api_call_count"],
    }
    return output


def call_tool_node(state: AgentState):
    print(":::NODE: call_tool:::")
    messages = state["messages"]
    last_message = messages[-1]
    tool = last_message.tool_calls[0]
    tool_name = tool["name"].lower()
    tool_args = tool["args"]
    tool_id = tool["id"]
    tool_output = tools[tool_name].invoke(tool_args)
    state["api_call_count"] += 1
    pprint(f"Tool output: {tool_output}")
    pprint(f"API call count after this tool call {state['api_call_count']}")
    tool_message = ToolMessage(
        content=tool_output,
        tool_call_id=tool_id,
    )
    output = {
        "messages": [tool_message],
        "api_call_count": state["api_call_count"],
    }
    return output


# %%

workflow = StateGraph(AgentState)
workflow.set_entry_point("agent")
workflow.add_node("agent", call_model_node)
workflow.add_node("action", call_tool_node)
workflow.add_conditional_edges(
    "agent",
    shoud_continue_edge,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")
workflow_compiled = workflow.compile()


# %%

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


# %%
messages = [
    SystemMessage(
        content=(
            "You are responsible for answering user questions. "
            "You use tools for that, These tools are available: "
            f"{', '.join(tools.keys())}.\n"
            "Use tools until you get the answer to the question. "
            "If you don't get the answer, try to call the tool again. "
            "You should not give up until you get the answer to the question. "
            "You can use tools multiple times if needed. "
            "You should not think about the answer, you should use tools. "
            "Reply only with the answer to the question. "
            "/no-think"
        )
    ),
    HumanMessage(
        content=("How is the weather in munich today? /no-think"),
    ),
]
started_state = {"messages": messages, "api_call_count": 0}
# started_state = AgentState(messages=messages, api_call_count=0)
result = workflow_compiled.invoke(started_state)
