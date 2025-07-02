from abc import ABC, abstractmethod
from enum import Enum

from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph
from loguru import logger


class IDisplayStrategy(ABC):
    def execute(self, workflow: StateGraph) -> None:
        pass


class ASCIIGraphDisplayStrategy(IDisplayStrategy):
    """
    This strategy using the ASCII to display the graph.
    """

    def execute(self, workflow: StateGraph) -> None:
        workflow.get_graph().print_ascii()


class MermaidGraphDisplayStrategy(IDisplayStrategy):
    """
    This strategy using the Mermaid API to display the graph.
    """

    def execute(self, workflow: StateGraph) -> None:
        display(
            Image(
                data=workflow.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                    max_retries=10,
                    retry_delay=2.0,
                ),
            ),
        )


class PyppeteerGraphDisplayStrategy(IDisplayStrategy):
    """
    This strategy using the Pyppeteer to display the graph in a browser.
    """

    def execute(self, workflow: StateGraph) -> None:
        display(
            Image(
                data=workflow.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.PYPPETEER,
                    max_retries=10,
                    retry_delay=2.0,
                ),
            ),
        )


class PNGGraphDisplayStrategy(IDisplayStrategy):
    """
    This strategy using the PNG to display the graph.
    """

    def execute(self, workflow: StateGraph) -> None:
        display(Image(workflow.get_graph().draw_png()))


# %%
class DisplayMethod(Enum):
    ASCII = ASCIIGraphDisplayStrategy()
    MERMAID = MermaidGraphDisplayStrategy()
    PYPPETEER = PyppeteerGraphDisplayStrategy()
    PNG = PNGGraphDisplayStrategy()


# %%
class IGraphDisplayer(ABC):
    @abstractmethod
    def set_strategy(self, display_method: DisplayMethod) -> "IGraphDisplayer":
        pass

    @abstractmethod
    def display(self) -> None:
        pass


class GraphDisplayer(IGraphDisplayer):
    _default_display_method = DisplayMethod.ASCII

    def __init__(self, workflow: StateGraph, display_method: DisplayMethod = None):
        self._workflow = workflow
        self._strategy = display_method or self._default_display_method.value

    def set_strategy(self, display_method: DisplayMethod) -> IGraphDisplayer:
        self._strategy = display_method.value
        return self

    def display(self) -> None:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        try:
            self._strategy.execute(self._workflow)
        except Exception as e:
            lg.error(f"error: {e}")
        lg.trace("end")


# %%
if __name__ == "__main__":
    from typing import TypedDict

    from langchain.chat_models import ChatOpenAI
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph import END, START, StateGraph

    from relepaper.domains.langgraph.interfaces import IWorkflowBuilder, IWorkflowNode

    class ExampleWorkflowState(TypedDict):
        value1: int = 0
        value2: int = 0
        value3: int = 0

    class Node1(IWorkflowNode):
        def __init__(self, llm: BaseChatModel):
            self._llm = llm

        def __call__(self, state: ExampleWorkflowState) -> ExampleWorkflowState:
            return {"value1": state["value1"] + 1, "value2": state["value2"]}

    class Node2(IWorkflowNode):
        def __init__(self, llm: BaseChatModel):
            self._llm = llm

        def __call__(self, state: ExampleWorkflowState) -> ExampleWorkflowState:
            return {"value1": state["value1"], "value2": state["value2"] + 1}

    class Node3(IWorkflowNode):
        def __init__(self, llm: BaseChatModel):
            self._llm = llm

        def __call__(self, state: ExampleWorkflowState) -> ExampleWorkflowState:
            return {"value1": state["value1"], "value2": state["value2"], "value3": state["value3"] + 1}

    class ExampleWorkflowBuilder(IWorkflowBuilder):
        def __init__(self, llm: BaseChatModel):
            self._llm = llm
            self._node1 = Node1(llm=llm)
            self._node2 = Node2(llm=llm)
            self._node3 = Node3(llm=llm)

        def build(self, **kwargs) -> StateGraph:
            lg = logger.bind(classname=self.__class__.__name__)
            lg.trace("start")
            builder = StateGraph(ExampleWorkflowState)
            builder.add_node("node1", self._node1)
            builder.add_node("node2", self._node2)
            builder.add_node("node3", self._node3)
            builder.add_edge(START, "node1")
            builder.add_edge("node1", "node2")
            builder.add_edge("node2", "node3")
            builder.add_edge("node3", END)
            compiled_graph = builder.compile(**kwargs)
            lg.trace("done")
            return compiled_graph

    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.0,
    )

    workflow = ExampleWorkflowBuilder(llm=llm).build()
    displayer = GraphDisplayer(workflow)
    displayer.display()

    displayer.set_strategy(DisplayMethod.MERMAID)
    displayer.display()

    displayer.set_strategy(DisplayMethod.PYPPETEER)
    displayer.display()

    displayer.set_strategy(DisplayMethod.PNG)
    displayer.display()
