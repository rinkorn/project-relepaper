from abc import ABC, abstractmethod
from typing import Any

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


class IStrategy(ABC):
    @abstractmethod
    def __call__(self, state: dict) -> Any:
        """Call the strategy."""
        pass


class IFactory(ABC):
    @abstractmethod
    def create(self, name: str) -> Any:
        """Create a instance of the class."""
        pass
