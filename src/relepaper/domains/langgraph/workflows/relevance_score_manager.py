# %%
import logging
import uuid
from pprint import pprint
from typing import List, TypedDict

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from relepaper.domains.langgraph.workflows.interfaces import IWorkflowBuilder, IWorkflowNode
from relepaper.domains.langgraph.workflows.utils import display_graph

__all__ = [
    "RelevanceScoreManagerWorkflowBuilder",
    "RelevanceScoreManagerState",
]

# %%
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    formatter = logging.Formatter("__log__: %(message)s")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# %%
def get_response_schemas() -> List[ResponseSchema]:
    return [
        ResponseSchema(
            name="decision",
            description="The decision to include the article in the dissertation",
            type="string",
            enum=["good score", "bad score"],
        ),
    ]


def get_prompt_template() -> str:
    return (
        "You are a helpful assistant that can help me to manage relevance scores. "
        "Accept the decision to include the article in the dissertation. "
        "If the mean relevance score is above 50, then answer 'good score', otherwise answer 'bad score'. "
        "MEAN RELEVANCE SCORE: {mean_relevance_score}"
        "FORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
    )


class RelevanceScoreManagerState(TypedDict):
    relevance_scores: List[float]
    mean_relevance_score: float
    decision: str


class RelevanceScoreManager0Node(IWorkflowNode):
    def __init__(self):
        pass

    def __call__(self, state: RelevanceScoreManagerState) -> RelevanceScoreManagerState:
        logger.info(":::NODE: RelevanceScoreManager0:::")
        relevance_scores = state["relevance_scores"]
        mean_relevance_score = sum(relevance_scores) / len(relevance_scores)

        if mean_relevance_score > 50:
            decision = "good score"
        else:
            decision = "bad score"

        output = {
            "mean_relevance_score": mean_relevance_score,
            "decision": decision,
        }
        return output


class RelevanceScoreManagerNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: RelevanceScoreManagerState) -> RelevanceScoreManagerState:
        logger.info(":::NODE: relevance_score_manager:::")
        relevance_scores = state["relevance_scores"]
        mean_relevance_score = sum(relevance_scores) / len(relevance_scores)

        prompt_template = get_prompt_template()
        response_schemas = get_response_schemas()
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["mean_relevance_score", "format_instructions"],
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self._llm | output_parser

        response = chain.invoke(
            {
                "mean_relevance_score": mean_relevance_score,
            }
        )
        pprint(response)
        output = {
            "mean_relevance_score": mean_relevance_score,
            "decision": response["decision"],
        }
        return output


class RelevanceScoreManagerWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def build(self, **kwargs) -> StateGraph:
        graph_builder = StateGraph(RelevanceScoreManagerState)
        # graph_builder.add_node("RelevanceScoreManager", RelevanceScoreManagerNode(llm=self._llm))
        graph_builder.add_node("RelevanceScoreManager0", RelevanceScoreManager0Node())
        # graph_builder.add_edge(START, "RelevanceScoreManager")
        # graph_builder.add_edge("RelevanceScoreManager", "RelevanceScoreManager0")
        graph_builder.add_edge(START, "RelevanceScoreManager0")
        graph_builder.add_edge("RelevanceScoreManager0", END)
        graph = graph_builder.compile(**kwargs)
        return graph


if __name__ == "__main__":
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
    workflow = RelevanceScoreManagerWorkflowBuilder(
        llm=llm,
    ).build(
        checkpointer=InMemorySaver(),
    )
    display_graph(workflow)

    # relevance_scores = [10, 20, 30, 40, 50]
    # relevance_scores = [40, 40, 40, 40, 40]
    relevance_scores = [50, 60, 70, 80, 90]
    state_start = RelevanceScoreManagerState(
        relevance_scores=relevance_scores,
    )
    config = {
        "configurable": {
            "max_concurrency": 4,
            "max_retries": 5,
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
