# %%
import logging
import operator
import os
import uuid
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
from langgraph.graph import END, START, StateGraph
from pydantic import Field

logger = logging.getLogger(__name__)


# %%
os.environ["OLLAMA_HOST"] = "http://10.0.0.14:11434"

llm = ChatOllama(
    model="qwen3:8b",
    # model="qwen3:14b",
    # model="qwen3:30B-a3b",
    # model="qwen3:32b",
    # model="qwen2.5-coder:14b",
    temperature=0.0,
    max_tokens=10000,
)


# %%
def reformulate_query(
    llm: BaseChatModel,
    query: str,
    quantity: int = 10,
    system_message: SystemMessage | str | None = None,
) -> str:
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
        HumanMessage(content=(f"{query}")),
    ]
    json_schema = {
        "description": "List of reformulated queries",
        "title": "query_variant_list",
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
tools = {}
tools["reformulate_query"] = tool_decorator(reformulate_query)
# llm_with_tools = llm.bind_tools([t for t in tools.values()], tool_choice="any")


# %%
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    reformulated_queries: List[str] = Field(description="List of reformulated queries")
    user_queries: Annotated[List[str], operator.add]


def reformulate_query_node(state: State):
    pprint(":::NODE: reformulate_query:::")
    original_query = state["messages"][-1].content
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
        "user_queries": [original_query] + reformulated_queries,
        "messages": [function_message],
    }
    return output


workflow = StateGraph(State)
workflow.set_entry_point("reformulate_query")

workflow.add_node("reformulate_query", reformulate_query_node)

workflow.add_edge(START, "reformulate_query")
workflow.add_edge("reformulate_query", END)

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
        "messages": started_messages,
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
class QueryInterpretatorService:
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
        state = self.graph.invoke(
            input=self.state,
            config=self.config,
        )
        pprint(state)
        return state

    def get_state_history(self):
        return self.graph.get_state_history(self.config)


if __name__ == "__main__":
    prompt = SystemMessage(
        content=(
            "You are responsible for answering user questions. "
            "You use tools for that, These tools are available: "
            f"{', '.join(tools.keys())}.\n"
        )
    )
    agent = QueryInterpretatorService(
        llm=llm,
        tools=tools,
        prompt=prompt,
        graph=workflow,
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
