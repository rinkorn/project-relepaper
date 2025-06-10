from abc import ABC, abstractmethod

from langgraph.graph import StateGraph


class IWorkflowBuilder(ABC):
    @abstractmethod
    def build(self, **kwargs) -> StateGraph:
        """Build the workflow with llm."""
        pass


class IWorkflowNode(ABC):
    @abstractmethod
    def __call__(self, state: dict) -> dict:
        """Call the workflow node."""
        pass


class IWorkflowEdge(ABC):
    @abstractmethod
    def __call__(self, state: dict) -> dict:
        """Call the workflow edge."""
        pass
