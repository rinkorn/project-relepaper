# %%
import uuid
from typing import List, TypedDict

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from relepaper.domains.langgraph.entities.relevance_decision import RelevanceDecision
from relepaper.domains.langgraph.entities.relevance_score import RelevanceScore
from relepaper.domains.langgraph.entities.relevance_score_container import RelevanceScoreContainer
from relepaper.domains.langgraph.workflows.interfaces import IWorkflowBuilder, IWorkflowNode
from relepaper.domains.langgraph.workflows.utils import display_graph

__all__ = [
    "RelevanceScoreManagerState",
    "RelevanceScoreManagerWorkflowBuilder",
]


# %%
def get_response_schemas() -> List[ResponseSchema]:
    return [
        ResponseSchema(
            name="decision",
            description="The decision to include the article to the research paper",
            type="string",
            enum=["good score", "bad score"],
        ),
    ]


def get_prompt_template() -> str:
    return (
        "You are a helpful assistant that can help me to manage relevance scores. "
        "Accept the decision to include the article to the research paper. "
        "If the mean score overall pdfs is above 50, then answer 'good score', otherwise answer 'bad score'. "
        "MEAN SCORE OVERALL PDFS: {mean_score_overall_pdfs}"
        "FORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
    )


class RelevanceScoreManagerState(TypedDict):
    relevance_scores: List[RelevanceScoreContainer]
    mean_score_overall_pdfs: float
    decision: RelevanceDecision


class RelevanceScoreManager0Node(IWorkflowNode):
    def __call__(self, state: RelevanceScoreManagerState) -> RelevanceScoreManagerState:
        logger.trace(f"{self.__class__.__name__}: __call__: start")
        relevance_scores = state["relevance_scores"]

        mean_scores = [container.mean for container in relevance_scores]
        mean_score = sum(mean_scores) / len(mean_scores)
        logger.debug(f"mean_scores: {mean_score}")
        logger.debug(f"mean score overall pdfs: {mean_score:.2f}")

        if mean_score > 50:
            decision = RelevanceDecision(decision="good score", comment="good score is above 50")
        else:
            decision = RelevanceDecision(decision="bad score", comment="bad score is below 50")

        logger.info(f"{self.__class__.__name__}: decision: {decision}")

        output = {
            "mean_score_overall_pdfs": mean_score,
            "decision": decision,
        }
        logger.trace(f"{self.__class__.__name__}: __call__: end")
        return output


class RelevanceScoreManagerNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: RelevanceScoreManagerState) -> RelevanceScoreManagerState:
        logger.trace(f"{self.__class__.__name__}: __call__: start")
        relevance_scores = state["relevance_scores"]

        mean_scores = [container.mean for container in relevance_scores]
        mean_score = sum(mean_scores) / len(mean_scores)
        logger.debug(f"mean scores by pdfs: {mean_scores}")
        logger.debug(f"mean score overall pdfs: {mean_score:.2f}")

        prompt_template = get_prompt_template()
        response_schemas = get_response_schemas()
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["mean_score_overall_pdfs"],
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self._llm | output_parser

        response = chain.invoke(
            {
                "mean_score_overall_pdfs": mean_score,
            }
        )
        decision = RelevanceDecision(decision=response["decision"])
        logger.info(f"{self.__class__.__name__}: decision: {decision}")
        output = {
            "mean_score_overall_pdfs": mean_score,
            "decision": decision,
        }
        logger.trace(f"{self.__class__.__name__}: __call__: end")
        return output


class RelevanceScoreManagerWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def build(self, **kwargs) -> StateGraph:
        logger.trace(f"{self.__class__.__name__}: build: start")
        graph_builder = StateGraph(RelevanceScoreManagerState)
        # graph_builder.add_node("RelevanceScoreManager", RelevanceScoreManager0Node())
        graph_builder.add_node("RelevanceScoreManager", RelevanceScoreManagerNode(llm=self._llm))
        graph_builder.add_edge(START, "RelevanceScoreManager")
        graph_builder.add_edge("RelevanceScoreManager", END)
        graph = graph_builder.compile(**kwargs)
        logger.trace(f"{self.__class__.__name__}: build: end")
        return graph


if __name__ == "__main__":
    from relepaper.config.logger import setup_logger

    setup_logger(stream_level="TRACE")
    # import os
    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # llm = ChatOllama(
    #     model="qwen3:8b",
    #     temperature=0.0,
    #     max_tokens=10000,
    # )
    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.00,
    )
    workflow = RelevanceScoreManagerWorkflowBuilder(llm=llm).build(checkpointer=InMemorySaver())
    display_graph(workflow)

    relevance_scores = [
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria="theme_score", score=25, comment=""),
                RelevanceScore(criteria="terminology_score", score=45, comment=""),
                RelevanceScore(criteria="methodology_score", score=50, comment=""),
                RelevanceScore(criteria="practical_applicability_score", score=25, comment=""),
                RelevanceScore(criteria="novelty_and_relevance_score", score=25, comment=""),
                RelevanceScore(criteria="fundamental_significance_score", score=25, comment=""),
            ]
        ),
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria="theme_score", score=15, comment=""),
                RelevanceScore(criteria="terminology_score", score=10, comment=""),
                RelevanceScore(criteria="methodology_score", score=10, comment=""),
                RelevanceScore(criteria="practical_applicability_score", score=15, comment=""),
                RelevanceScore(criteria="novelty_and_relevance_score", score=10, comment=""),
                RelevanceScore(criteria="fundamental_significance_score", score=10, comment=""),
            ]
        ),
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria="theme_score", score=20, comment=""),
                RelevanceScore(criteria="terminology_score", score=45, comment=""),
                RelevanceScore(criteria="methodology_score", score=20, comment=""),
                RelevanceScore(criteria="practical_applicability_score", score=20, comment=""),
                RelevanceScore(criteria="novelty_and_relevance_score", score=20, comment=""),
                RelevanceScore(criteria="fundamental_significance_score", score=20, comment=""),
            ]
        ),
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria="theme_score", score=90, comment=""),
                RelevanceScore(criteria="terminology_score", score=80, comment=""),
                RelevanceScore(criteria="methodology_score", score=85, comment=""),
                RelevanceScore(criteria="practical_applicability_score", score=90, comment=""),
                RelevanceScore(criteria="novelty_and_relevance_score", score=90, comment=""),
                RelevanceScore(criteria="fundamental_significance_score", score=80, comment=""),
            ]
        ),
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria="theme_score", score=90, comment=""),
                RelevanceScore(criteria="terminology_score", score=80, comment=""),
                RelevanceScore(criteria="methodology_score", score=85, comment=""),
                RelevanceScore(criteria="practical_applicability_score", score=90, comment=""),
                RelevanceScore(criteria="novelty_and_relevance_score", score=90, comment=""),
                RelevanceScore(criteria="fundamental_significance_score", score=80, comment=""),
            ]
        ),
        RelevanceScoreContainer(
            scores=[
                RelevanceScore(criteria="theme_score", score=50, comment=""),
                RelevanceScore(criteria="terminology_score", score=60, comment=""),
                RelevanceScore(criteria="methodology_score", score=70, comment=""),
                RelevanceScore(criteria="practical_applicability_score", score=40, comment=""),
                RelevanceScore(criteria="novelty_and_relevance_score", score=80, comment=""),
                RelevanceScore(criteria="fundamental_significance_score", score=65, comment=""),
            ]
        ),
    ]

    state_start = RelevanceScoreManagerState(
        relevance_scores=relevance_scores[:3],
        mean_score_overall_pdfs=None,
        decision=None,
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
    print(state_end)

    # state_history = [sh for sh in workflow.get_state_history(config)]
    # pprint(state_history)
