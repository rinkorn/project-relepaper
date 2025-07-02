# %%
import uuid
from typing import List, TypedDict

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from relepaper.domains.langgraph.entities.relevance_decision import RelevanceDecision, RelevanceStatus, Threshold
from relepaper.domains.langgraph.entities.relevance_score import RelevanceCriteria, RelevanceScore, Score
from relepaper.domains.langgraph.entities.relevance_score_container import RelevanceScoreContainer
from relepaper.domains.langgraph.interfaces import IStrategy, IWorkflowBuilder, IWorkflowNode

__all__ = [
    "RelevanceManagerState",
    "RelevanceManagerWorkflowBuilder",
]


# %%
def get_response_schemas() -> List[ResponseSchema]:
    return [
        ResponseSchema(
            name="decision",
            description="The decision to include the article to the research paper",
            type="string",
            enum=[RelevanceStatus.RELEVANT.value, RelevanceStatus.NOT_RELEVANT.value],
        ),
    ]


def get_prompt_template() -> str:
    return (
        "You are a helpful assistant that can help me to manage relevance scores. "
        "Accept the decision to include the article to the research paper. "
        "If the mean score overall pdfs is above or equal to {decision_threshold}, then answer 'RELEVANT', otherwise answer 'NOT_RELEVANT'. "
        "MEAN SCORE OVERALL PDFS: {mean_score_overall_pdfs}"
        "FORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
    )


# %%
class RelevanceManagerState(TypedDict):
    relevance_scores: List[RelevanceScoreContainer]
    mean_score_overall_pdfs: float
    decision_threshold: Threshold
    relevance_decision: RelevanceDecision


# %%
class RelevanceThresholdDecisionStrategy(IStrategy):
    def __call__(self, state: RelevanceManagerState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        relevance_scores = state["relevance_scores"]
        decision_threshold = state["decision_threshold"]

        mean_scores = [container.mean for container in relevance_scores]
        mean_score = sum(mean_scores) / len(mean_scores) if len(mean_scores) > 0 else 0
        lg.debug(f"mean_scores: {mean_scores}")
        lg.debug(f"mean score overall pdfs: {mean_score:.2f}")

        if mean_score >= decision_threshold.value:
            conclusion = RelevanceStatus.RELEVANT
        else:
            conclusion = RelevanceStatus.NOT_RELEVANT

        decision = RelevanceDecision(
            status=conclusion,
            comment=(
                "Simple threshold decision strategy made the decision. "
                f"The mean score overall pdfs is {mean_score:.2f} and the decision threshold is {decision_threshold.value:.2f}. "
                f"The conclusion is {conclusion.value}."
            ),
        )
        lg.info(f"conclusion: {decision.status}")
        output = {
            "mean_score_overall_pdfs": mean_score,
            "relevance_decision": decision,
        }
        lg.trace("end")
        return output


class RelevanceLLMDecisionStrategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: RelevanceManagerState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        relevance_scores = state["relevance_scores"]
        decision_threshold = state["decision_threshold"]
        mean_scores = [container.mean for container in relevance_scores]
        mean_score = sum(mean_scores) / len(mean_scores)
        lg.debug(f"mean scores by pdfs: {mean_scores}")
        lg.debug(f"mean score overall pdfs: {mean_score:.2f}")

        prompt_template = get_prompt_template()
        response_schemas = get_response_schemas()
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["mean_score_overall_pdfs", "decision_threshold"],
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self._llm | output_parser

        response = chain.invoke(
            {
                "mean_score_overall_pdfs": mean_score,
                "decision_threshold": decision_threshold.value,
            }
        )
        lg.info(f"response: {response}")
        decision = RelevanceDecision(
            status=RelevanceStatus(response["decision"]),
            comment="LLM made the decision by using the mean score overall pdfs and the decision threshold",
        )
        lg.info(f"conclusion: {decision.status}")
        output = {
            "mean_score_overall_pdfs": mean_score,
            "relevance_decision": decision,
        }
        lg.trace("end")
        return output


# %%
class RelevanceDecisionNode(IWorkflowNode):
    _default_decision_strategy: IStrategy = RelevanceThresholdDecisionStrategy()

    def __init__(self, strategy: IStrategy | None = None):
        self._decision_strategy = strategy or self._default_decision_strategy

    def set_strategy(self, strategy: IStrategy):
        self._decision_strategy = strategy
        return self

    def __call__(self, state: RelevanceManagerState) -> RelevanceManagerState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        lg.debug(f"decision_strategy: {self._decision_strategy.__class__.__name__}")

        state_output = self._decision_strategy(state)
        output = {
            "mean_score_overall_pdfs": state_output["mean_score_overall_pdfs"],
            "relevance_decision": state_output["relevance_decision"],
        }

        lg.trace("end")
        return output


# %%
class RelevanceManagerWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel, decision_strategy: IStrategy | None = None):
        self._llm = llm
        self._relevance_node = RelevanceDecisionNode(strategy=decision_strategy)

    def build(self, **kwargs) -> StateGraph:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        graph_builder = StateGraph(RelevanceManagerState)
        graph_builder.add_node("RelevanceManager", self._relevance_node)
        graph_builder.add_edge(START, "RelevanceManager")
        graph_builder.add_edge("RelevanceManager", END)
        graph = graph_builder.compile(**kwargs)
        lg.trace("end")
        return graph


if __name__ == "__main__":
    from relepaper.config.logger import setup_logger
    from relepaper.domains.langgraph.workflows.utils.graph_displayer import (
        DisplayMethod,
        GraphDisplayer,
    )

    setup_logger(stream_level="TRACE")
    # import os
    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # llm = ChatOllama(
    #     model="qwen3:8b",
    #     temperature=0.0,
    #     max_tokens=10000,
    # )
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.00,
    )
    workflow_builder = RelevanceManagerWorkflowBuilder(
        llm=llm,
        decision_strategy=RelevanceThresholdDecisionStrategy(),
        # decision_strategy=RelevanceLLMDecisionStrategy(llm=llm),
    )
    workflow = workflow_builder.build(checkpointer=InMemorySaver())
    displayer = GraphDisplayer(workflow).set_strategy(DisplayMethod.MERMAID)
    displayer.display()

    relevance_scores = [
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria=RelevanceCriteria.THEME, score=Score(25), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.TERMINOLOGY, score=Score(45), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.METHODOLOGY, score=Score(50), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.PRACTICAL_APPLICABILITY, score=Score(25), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.NOVELTY_AND_RELEVANCE, score=Score(25), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.FUNDAMENTAL_SIGNIFICANCE, score=Score(25), comment=""),
            ]
        ),
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria=RelevanceCriteria.THEME, score=Score(15), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.TERMINOLOGY, score=Score(10), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.METHODOLOGY, score=Score(10), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.PRACTICAL_APPLICABILITY, score=Score(15), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.NOVELTY_AND_RELEVANCE, score=Score(10), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.FUNDAMENTAL_SIGNIFICANCE, score=Score(10), comment=""),
            ]
        ),
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria=RelevanceCriteria.THEME, score=Score(20), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.TERMINOLOGY, score=Score(45), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.METHODOLOGY, score=Score(20), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.PRACTICAL_APPLICABILITY, score=Score(20), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.NOVELTY_AND_RELEVANCE, score=Score(20), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.FUNDAMENTAL_SIGNIFICANCE, score=Score(20), comment=""),
            ]
        ),
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria=RelevanceCriteria.THEME, score=Score(90), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.TERMINOLOGY, score=Score(80), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.METHODOLOGY, score=Score(85), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.PRACTICAL_APPLICABILITY, score=Score(90), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.NOVELTY_AND_RELEVANCE, score=Score(90), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.FUNDAMENTAL_SIGNIFICANCE, score=Score(80), comment=""),
            ]
        ),
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria=RelevanceCriteria.THEME, score=Score(90), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.TERMINOLOGY, score=Score(80), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.METHODOLOGY, score=Score(85), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.PRACTICAL_APPLICABILITY, score=Score(90), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.NOVELTY_AND_RELEVANCE, score=Score(90), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.FUNDAMENTAL_SIGNIFICANCE, score=Score(80), comment=""),
            ]
        ),
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria=RelevanceCriteria.THEME, score=Score(50), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.TERMINOLOGY, score=Score(60), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.METHODOLOGY, score=Score(70), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.PRACTICAL_APPLICABILITY, score=Score(40), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.NOVELTY_AND_RELEVANCE, score=Score(80), comment=""),
                RelevanceScore(criteria=RelevanceCriteria.FUNDAMENTAL_SIGNIFICANCE, score=Score(65), comment=""),
            ]
        ),
    ]

    state_start = RelevanceManagerState(
        relevance_scores=relevance_scores[:3],
        mean_score_overall_pdfs=None,
        decision_threshold=Threshold(50),
        relevance_decision=None,
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
    print(f"mean score overall pdfs: {state_end['mean_score_overall_pdfs']:.2f}")
    print(f"relevance_decision: {state_end['relevance_decision']}")

    # state_history = [sh for sh in workflow.get_state_history(config)]
    # pprint(state_history)
