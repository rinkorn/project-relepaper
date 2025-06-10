# %%
import json
import os
import uuid
from pathlib import Path
from typing import Annotated, Sequence, operator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from relepaper.domains.langgraph.entities.session import Session
from relepaper.domains.langgraph.workflows.interfaces import IWorkflowBuilder, IWorkflowEdge, IWorkflowNode
from relepaper.domains.langgraph.workflows.utils import display_graph
from relepaper.domains.openalex.entities.work import OpenAlexWork


# %%
class TypstManagerWorkflowState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session: Session
    works: list[OpenAlexWork]


class UserQueryNode(IWorkflowNode):
    def __init__(self):
        self._config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}

    def __call__(self, state: TypstManagerWorkflowState) -> TypstManagerWorkflowState:
        user_query = input("Enter your query: ")
        output = {
            "messages": [HumanMessage(content=user_query)],
        }
        return output


class UserQueryConditionalEdge(IWorkflowEdge):
    def __init__(self):
        self._config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}

    def __call__(self, state: TypstManagerWorkflowState) -> str:
        user_query = state["messages"][-1].content
        if user_query in ["exit", "quit", "stop", "end", "finish", "done", "close", "q"]:
            return "exit"
        return "continue"


# %%
class GenerateTypstDocumentNode(IWorkflowNode):
    def __init__(self):
        # self._agent = create_react_agent(llm, checkpointer=InMemorySaver())
        self._config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}

    def __call__(self, state: TypstManagerWorkflowState) -> TypstManagerWorkflowState:
        state["works"]

        # # print(f"generate_pdf_act({asdict(customer)}, {list(map(lambda j: asdict(j), jobs))})")
        # act_json = {
        #     "customer": asdict(customer),
        #     "jobs": list(
        #         map(lambda j: asdict(j), jobs),
        #     ),
        # }
        # with open(os.path.join("typst", "act.json"), "w") as f:
        #     json.dump(act_json, f, ensure_ascii=False)
        # command = ["typst", "compile", "--root", "./typst", "typst/act.typ"]
        # try:
        #     subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # except subprocess.CalledProcessError as e:
        #     print(e.stderr)

        output = {}
        return output


class RedactTypstDocumentNode(IWorkflowNode):
    def __init__(self):
        self._config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}

    def __call__(self, state: TypstManagerWorkflowState) -> TypstManagerWorkflowState:
        state["works"]
        # invoice_json = {
        #     "customer": asdict(customer),
        #     "jobs": list(
        #         map(lambda j: asdict(j), jobs),
        #     ),
        # }
        # with open(os.path.join("typst", "invoice.json"), "w") as f:
        #     json.dump(invoice_json, f, ensure_ascii=False)
        # command = ["typst", "compile", "--root", "./typst", "typst/invoice.typ"]
        # try:
        #     subprocess.run(
        #         command,
        #         check=True,
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.PIPE,
        #         text=True,
        #     )
        # except subprocess.CalledProcessError as e:
        #     print(e.stderr)

        output = {}
        return output


class TypstAgentNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}
        system_prompt = (
            "You are a helpful assistant that can generate typst documents. "
            "You will be given a list of works and you will need to generate a typst document. "
            "The document will be a typst document. "
        )
        self._agent = create_react_agent(
            llm,
            tools=[],
            checkpointer=InMemorySaver(),
            system_prompt=system_prompt,
        )

    def __call__(self, state: TypstManagerWorkflowState) -> TypstManagerWorkflowState:
        user_query = state["messages"][-1].content
        works = state["works"]
        state_input = {
            "messages": [HumanMessage(content=user_query)],
            "works": works,
        }
        state_output = self._agent.invoke(state_input)
        output = {
            "messages": state_output["messages"],
        }
        return output


class TypstAgentConditionalEdge(IWorkflowEdge):
    def __init__(self):
        self._config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}

    def __call__(self, state: TypstManagerWorkflowState) -> str:
        condition = "create"
        if condition == "create":
            return "create"
        elif condition == "redact":
            return "redact"
        else:
            return "exit"


# %%
class TypstManagerWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        # self._agent = create_react_agent(llm, tools=[], checkpointer=InMemorySaver())
        self._config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}

    def build(self, **kwargs) -> StateGraph:
        graph_builder = StateGraph(TypstManagerWorkflowState)
        graph_builder.add_node("UserQuery", UserQueryNode())
        graph_builder.add_node("TypstAgent", TypstAgentNode(self._llm))
        graph_builder.add_node("RedactTypstDocument", RedactTypstDocumentNode())
        graph_builder.add_node("GenerateTypstDocument", GenerateTypstDocumentNode())
        graph_builder.add_edge(START, "UserQuery")
        graph_builder.add_conditional_edges(
            "UserQuery",
            UserQueryConditionalEdge(),
            {
                "continue": "TypstAgent",
                "exit": END,
            },
        )
        graph_builder.add_conditional_edges(
            "TypstAgent",
            TypstAgentConditionalEdge(),
            {
                "create": "GenerateTypstDocument",
                "redact": "RedactTypstDocument",
                "exit": "UserQuery",
            },
        )
        graph_builder.add_edge("GenerateTypstDocument", "TypstAgent")
        graph_builder.add_edge("RedactTypstDocument", "TypstAgent")
        compiled_graph = graph_builder.compile()
        return compiled_graph


if __name__ == "__main__":
    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.0,
        max_tokens=10000,
    )
    workflow = TypstManagerWorkflowBuilder(llm).build()
    display_graph(workflow)

    session = Session()
    works_path = Path(".data") / "openalex_works"
    works = [
        OpenAlexWork.from_dict(json.load(open(work_id)))
        for work_id in works_path.iterdir()
        if work_id.is_file() and work_id.suffix == ".json"
    ]

    system_prompt = (
        "You are a helpful assistant that can generate typst documents. "
        "You will be given a list of works and you will need to generate a typst document. "
        "The document will be a typst document. "
    )

    state_start = TypstManagerWorkflowState(
        messages=[SystemMessage(content=system_prompt)],
        session=session,
        works=works,
    )
    state_end = workflow.invoke(input=state_start)
    print(state_end)

# %%
