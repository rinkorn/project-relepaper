from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph


def display_graph(workflow: StateGraph) -> None:
    display(
        Image(
            data=workflow.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            ),
        ),
    )
