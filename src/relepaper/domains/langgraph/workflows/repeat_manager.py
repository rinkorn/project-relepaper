# %%
import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from relepaper.domains.langgraph.entities.relevance_decision import RelevanceDecision, RelevanceStatus
from relepaper.domains.langgraph.entities.repeat_decision import RepeatStatus
from relepaper.domains.langgraph.entities.session import Session
from relepaper.domains.langgraph.interfaces import IStrategy, IWorkflowBuilder, IWorkflowNode

__all__ = [
    "RepeaterState",
    "RepeaterWorkflowBuilder",
    "NotRelevantAndRepeatDecisionStrategy",
    "StopDecisionStrategy",
]


# %%
class RepeaterState(TypedDict):
    session: Session
    relevance_decision: RelevanceDecision
    repeat_status: RepeatStatus
    max_repetitions: int = 3
    current_repetition: int = 1


# %%
class StopDecisionStrategy(IStrategy):
    def __call__(self, state: RepeaterState) -> RepeaterState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        repeat_status = RepeatStatus.STOP
        lg.info(f"repeat_status: {repeat_status}")
        lg.trace("end")
        output = {
            "repeat_status": repeat_status,
        }
        return output


class NotRelevantAndRepeatDecisionStrategy(IStrategy):
    def __call__(self, state: RepeaterState) -> RepeaterState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        relevance_decision = state["relevance_decision"]
        max_repetitions = state["max_repetitions"]
        current_repetition = state["current_repetition"]
        lg.debug(f"relevance_decision: {relevance_decision.status}")
        lg.debug(f"max_repetitions: {max_repetitions}")
        lg.debug(f"current_repetition before decision: {current_repetition}")
        if relevance_decision.status == RelevanceStatus.NOT_RELEVANT and current_repetition <= max_repetitions:
            repeat_status = RepeatStatus.REPEAT
            current_repetition += 1
        elif relevance_decision.status == RelevanceStatus.NOT_RELEVANT and current_repetition > max_repetitions:
            repeat_status = RepeatStatus.STOP
        elif relevance_decision.status == RelevanceStatus.RELEVANT:
            repeat_status = RepeatStatus.STOP
        else:
            lg.error(f"Invalid relevance decision: {relevance_decision.status}")
            repeat_status = RepeatStatus.STOP
        lg.debug(f"current_repetition after decision: {current_repetition}")
        lg.info(f"repeat_status: {repeat_status}")
        lg.trace("end")
        output = {
            "repeat_status": repeat_status,
            "current_repetition": current_repetition,
        }
        return output


# %%
class RepeaterNode(IWorkflowNode):
    """
    Accepts a decision to repeat or stop.
    """

    _default_strategy: IStrategy = StopDecisionStrategy()
    _strategy: IStrategy = _default_strategy

    def set_strategy(self, strategy: IStrategy | None = None):
        self._strategy = strategy if strategy else self._default_strategy
        return self

    def __call__(self, state: RepeaterState) -> RepeaterState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        lg.info(f"strategy: {self._strategy.__class__.__name__}")
        try:
            output = self._strategy(state)
        except Exception as e:
            lg.error(f"error: {e}")
            output = self._default_strategy(state)
        lg.trace("end")
        return output


# %%
class RepeaterWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, repeate_strategy: IStrategy | None = None):
        self._repeater_node = RepeaterNode().set_strategy(repeate_strategy)

    def build(self, **kwargs) -> StateGraph:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        graph_builder = StateGraph(RepeaterState)
        graph_builder.add_node("repeater", self._repeater_node)
        graph_builder.add_edge(START, "repeater")
        graph_builder.add_edge("repeater", END)
        graph = graph_builder.compile(**kwargs)
        lg.trace("end")
        return graph


if __name__ == "__main__":
    from relepaper.config.logger import setup_logger
    from relepaper.domains.langgraph.workflows.utils.graph_displayer import DisplayMethod, GraphDisplayer

    setup_logger(stream_level="DEBUG")

    workflow_builder = RepeaterWorkflowBuilder()
    workflow = workflow_builder.build(checkpointer=InMemorySaver())
    displayer = GraphDisplayer(workflow).set_strategy(DisplayMethod.MERMAID)
    displayer.display()

    state_start = RepeaterState(
        relevance_decision=RelevanceDecision(RelevanceStatus.NOT_RELEVANT),
        repeat_status=None,
        current_repetition=1,
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
    print(f"repeat_status: {state_end['repeat_status']}")
    print(f"current_repetition: {state_end['current_repetition']}")
    print(f"max_repetitions: {state_end['max_repetitions']}")

    # state_history = [sh for sh in workflow.get_state_history(config)]
    # pprint(state_history)
