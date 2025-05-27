# %%
import logging
from pprint import pprint

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool as tool_decorator
from langchain_ollama import ChatOllama


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
