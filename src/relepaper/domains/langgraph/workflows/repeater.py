# %%
import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from relepaper.domains.langgraph.entities.relevance_decision import RelevanceDecision, RelevanceStatus
from relepaper.domains.langgraph.entities.repeat_decision import RepeatStatus
from relepaper.domains.langgraph.interfaces import IWorkflowBuilder, IWorkflowEdge, IWorkflowNode

__all__ = [
    "RepeaterState",
    "RepeaterWorkflowBuilder",
]


# %%
class RepeaterState(TypedDict):
    relevance_decision: RelevanceDecision
    repeat_state: RepeatStatus
    max_repetitions: int = 3
    current_repetition: int = 0


# %%
class RepeaterDecisionConditionEdge(IWorkflowEdge):
    def __call__(self, state: RepeaterState) -> RepeatStatus:
        logger.trace(f"{self.__class__.__name__}: __call__: start")
        relevance_decision = state["relevance_decision"]
        max_repetitions = state["max_repetitions"]
        current_repetition = state["current_repetition"]
        logger.debug(f"{self.__class__.__name__}: __call__: relevance_decision: {relevance_decision.status}")
        logger.debug(f"{self.__class__.__name__}: __call__: max_repetitions: {max_repetitions}")
        logger.debug(f"{self.__class__.__name__}: __call__: current_repetition: {current_repetition}")
        if relevance_decision.status == RelevanceStatus.NOT_RELEVANT and current_repetition < max_repetitions:
            repeat_state = RepeatStatus.REPEAT
            state["current_repetition"] += 1
            logger.debug(f"{self.__class__.__name__}: __call__: repeat_state: {repeat_state}")
            logger.debug(f"{self.__class__.__name__}: __call__: current_repetition: {state['current_repetition']}")
        elif relevance_decision.status == RelevanceStatus.RELEVANT:
            repeat_state = RepeatStatus.STOP
            logger.debug(f"{self.__class__.__name__}: __call__: repeat_state: {repeat_state}")
        else:
            logger.error(f"Invalid relevance decision: {relevance_decision.status}")
            raise ValueError(f"Invalid relevance decision: {relevance_decision.status}")

        return repeat_state


# %%
class EmptyNode(IWorkflowNode):
    def __call__(self, state: RepeaterState) -> RepeaterState:
        return state


# %%
class RepeaterWorkflowBuilder(IWorkflowBuilder):
    def build(self, **kwargs) -> StateGraph:
        logger.trace(f"{self.__class__.__name__}: build: start")
        graph_builder = StateGraph(RepeaterState)
        graph_builder.add_node("EmptyNode", EmptyNode())
        graph_builder.add_conditional_edges(
            START,
            RepeaterDecisionConditionEdge(),
            {
                RepeatStatus.STOP: END,
                RepeatStatus.REPEAT: "EmptyNode",
            },
        )
        graph = graph_builder.compile(**kwargs)
        logger.trace(f"{self.__class__.__name__}: build: end")
        return graph


if __name__ == "__main__":
    from relepaper.config.logger import setup_logger

    setup_logger(stream_level="TRACE")

    workflow_builder = RepeaterWorkflowBuilder()
    workflow = workflow_builder.build(checkpointer=InMemorySaver())

    state_start = RepeaterState(
        relevance_decision=RelevanceDecision(
            status=RelevanceStatus.NOT_RELEVANT,
            comment="simple comment about the decision",
        ),
        repeat_state=None,
        current_repetition=0,
        max_repetitions=3,
    )
    config = {
        "configurable": {
            "max_retries": 5,
            "max_concurrency": 1,
            "thread_id": uuid.uuid4().hex,
        },
    }
    state_end = workflow.invoke(
        input=state_start,
        config=config,
    )
    print(f"repeat_state: {state_end['repeat_state']}")
    print(f"current_repetition: {state_end['current_repetition']}")
    print(f"max_repetitions: {state_end['max_repetitions']}")

    # state_history = [sh for sh in workflow.get_state_history(config)]
    # pprint(state_history)
