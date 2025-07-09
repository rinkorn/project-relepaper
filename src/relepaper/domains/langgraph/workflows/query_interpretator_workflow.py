# %%
import uuid
from enum import Enum
from pprint import pprint
from typing import List, TypedDict

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel, Field

from relepaper.domains.langgraph.entities.session import Session
from relepaper.domains.langgraph.interfaces import IStrategy, IWorkflowBuilder, IWorkflowNode

__all__ = [
    "QueryInterpretatorState",
    "QueryInterpretatorWorkflowBuilder",
]


# %%
class QueryInterpretatorState(TypedDict):
    session: Session
    user_query: BaseMessage
    main_topic: str
    context_for_queries: str
    reformulated_queries_quantity: int = 10
    reformulated_queries: List[str]
    comment: str | None = None


# %%
def get_context_maker_system_message(previous_context: str = "", think: bool = True):
    message = (
        "You are an expert in scientific research and bibliometric analysis. "  # "Ты эксперт по научным исследованиям и библиометрическому анализу. "
        "Your task is to analyze the user query and create a detailed context "  # "Твоя задача - проанализировать пользовательский запрос, выделить главную тему и создать детальный контекст "
        "for finding relevant scientific articles.\n\n"  # "для поиска релевантных научных статей в Google Scholar.\n\n"
        "You no need download any articles. "  # "Ты не нужно скачивать статьи. "
        "You only need to analyze the user query and create a detailed context "  # "Ты только нужно проанализировать пользовательский запрос и создать детальный контекст "
        "for finding relevant scientific articles.\n\n"  # "для поиска релевантных научных статей в Google Scholar.\n\n"
        "ANALYZE the user query and DETERMINE:\n"  # "АНАЛИЗИРУЙ запрос пользователя и ОПРЕДЕЛИ:\n"
        "1. Main topic of the user query. "  # "1. Главную тему исследования\n"
        "2. Main scientific field and discipline\n"  # "2. Основную научную область и дисциплину\n"
        "3. Key terms and concepts (including synonyms and ENGLISH equivalents)\n"  # "3. Ключевые термины и понятия (включая синонимы и англоязычные эквиваленты)\n"
        "4. Research methods that may be relevant\n"  # "4. Методы исследования, которые могут быть релевантны\n"
        "5. Related areas of knowledge\n"  # "5. Смежные области знаний\n"
        "6. Temporal context of research (if specified)\n"  # "6. Временной контекст исследований (если указан)\n"
        "7. Level of research (theoretical, experimental, applied)\n\n"  # "7. Уровень исследования (теоретический, экспериментальный, прикладной)\n\n"
        "CREATE a structured context, including:\n"  # "СОЗДАЙ структурированный контекст, включающий:\n"
        "- Alternative formulations of terms\n"  # "- Альтернативные формулировки терминов\n"
        "- Related concepts and areas\n"  # "- Смежные понятия и области\n"
        "- Potential methodological approaches\n"  # "- Потенциальные методологические подходы\n"
        "- Specific technologies or equipment (if mentioned)\n\n"  # "- Специфические технологии или оборудование (если упомянуты)\n\n"
        "Use ENGLISH for all responses.\n"  # "Используй РУССКИЙ язык для всех ответов.\n"
        "The context should be detailed enough to create a variety of search queries, "  # "Контекст должен быть достаточно подробным для создания разнообразных поисковых запросов, "
        "but focused on the main topic of the research."  # "но сфокусированным на основной теме исследования. "
    )
    if not think:
        message += "\n/no-think\n"
    if previous_context:
        message += (
            "\n\nBelow is a context that did not work well last time. "  # "\n\nНиже приводится контекст, который плохо сработал в прошлый раз. "
            "Improve it: expand, clarify, add new topics, remove topics that are not relevant. "  # "Его необходимо улучшить: расширить, уточнить, добавить новые темы, убрать темы которые не подходят. "
            f"\nPREVIOUS CONTEXT: \n{previous_context}\n\n"  # f"\nPrevious context: {previous_context}\n\n"
        )
    return message


# %%
def get_structured_output_schema():
    return {
        "title": "Queries main topic and context with comment",
        "type": "object",
        "properties": {
            "comment": {
                "type": "string",
                "description": "Comment to the user query. "
                "It should be a short comment that captures the core idea of the user query."
                "Use English for all responses.",
            },
            "main_topic": {
                "type": "string",
                "description": "Main topic of the user query. "
                "It should be a single word or phrase that captures the core idea of the user query."
                "Use English for all responses.",
            },
            "context_for_queries": {
                "type": "string",
                "description": "Extremely detailed and comprehensive academic review of the main topic. "
                "This should be a thorough scholarly analysis covering all aspects of the field including: "
                "theoretical foundations, methodologies, current research, applications, challenges, "
                "key researchers, terminology variations, and future directions. "
                "Write as if creating a comprehensive literature review. "
                "Minimum 2000-3000 words. Use rich academic language and include specific technical details. "
                "Use English for all responses.",
            },
        },
        "required": ["comment", "main_topic", "context_for_queries"],
    }


class WithStructuredOutputStrategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._config = {
            "configurable": {
                "thread_id": uuid.uuid4().hex,
            },
        }

    def __call__(self, state: QueryInterpretatorState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")

        user_query = state.get("user_query")
        lg.debug(f"User query: {user_query.content}")

        previous_context = state.get("context_for_queries")
        lg.debug(f"Previous context: {previous_context}")

        system_message = get_context_maker_system_message(
            think=False,
            previous_context=previous_context,
        )
        # lg.debug(f"System message: {system_message}")

        json_schema = get_structured_output_schema()
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=state.get("user_query").content),
        ]
        structured_llm = self._llm.with_structured_output(
            schema=json_schema,
            method="json_schema",
        )
        response = structured_llm.invoke(
            messages,
            config=self._config,
        )
        output = {
            "comment": response.get("comment"),
            "main_topic": response.get("main_topic"),
            "context_for_queries": response.get("context_for_queries"),
        }
        lg.trace("end")
        return output


if __name__ == "__main__":
    import os

    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:32b",
        temperature=0.0,
        max_tokens=10000,
    )
    # from langchain.chat_models import ChatOpenAI

    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.00,
    # )
    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    previous_context = (
        "Reinforcement Learning (RL) - is a field of machine learning where an agent learns "
        "to make decisions by interacting with an environment to maximize a reward. "
        "The main concepts include: agent, environment, actions, rewards, policies, value functions, "
        "and learning algorithms. "
        "The main algorithms include Q-learning, SARSA, Deep Q-Networks (DQN), Policy Gradients, "
        "Actor-Critic methods, and modern approaches such as Proximal Policy Optimization (PPO), "
        "Trust Region Policy Optimization (TRPO), and tensor-based algorithms such as Deep Deterministic "
        "Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3)."
    )
    state_start_with_structured_output = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=previous_context,
        reformulated_queries_quantity=2,
    )
    state_end_with_structured_output = WithStructuredOutputStrategy(llm)(state_start_with_structured_output)
    print("--------------------------------")
    print("User query:")
    pprint(user_query.content)
    print("--------------------------------")
    print("Comment:")
    pprint(state_end_with_structured_output.get("comment"))
    print("--------------------------------")
    print("Main topic:")
    pprint(state_end_with_structured_output.get("main_topic"))
    print("--------------------------------")
    print("Context for queries:")
    pprint(state_end_with_structured_output.get("context_for_queries"))
    print("--------------------------------")


# %%
class AgentWithResponseFormatStrategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: QueryInterpretatorState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.info("start")

        user_query = state.get("user_query")
        lg.debug(f"User query: {user_query.content}")

        previous_context = state.get("context_for_queries")
        lg.debug(f"Previous context: {previous_context}")

        system_message = get_context_maker_system_message(
            think=False,
            previous_context=previous_context,
        )
        # lg.debug(f"System message: {system_message}")

        class AnswerResponseFormat(BaseModel):
            comment: str = Field(description="Comment to the user query")
            main_topic: str = Field(description="Main topic of the user query")
            summary_article: str = Field(
                description=(
                    "Summary long article about the main topic. "
                    "It should be a summary of the main topic and the most important information about it. "
                    "1000-5000 words."
                )
            )

        self._agent = create_react_agent(
            model=llm,
            tools=[],
            prompt=system_message,
            name="context_maker",
            response_format=AnswerResponseFormat,
            # checkpointer=InMemorySaver(),
        )
        self._config = {
            "configurable": {
                "thread_id": uuid.uuid4().hex,
            },
        }

        response = self._agent.invoke(
            input={"messages": [user_query]},
            config=self._config,
        )["structured_response"].__dict__
        output = {
            "comment": response["comment"],
            "main_topic": response["main_topic"],
            "context_for_queries": response["summary_article"],
        }
        lg.info("end")
        return output


if __name__ == "__main__":
    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:32b",
        temperature=0.0,
        max_tokens=10000,
    )

    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    previous_context = (
        "Reinforcement Learning (RL) - is a field of machine learning where an agent learns "
        "to make decisions by interacting with an environment to maximize a reward. "
        "The main concepts include: agent, environment, actions, rewards, policies, value functions, "
        "and learning algorithms. "
        "The main algorithms include Q-learning, SARSA, Deep Q-Networks (DQN), Policy Gradients, "
        "Actor-Critic methods, and modern approaches such as Proximal Policy Optimization (PPO), "
        "Trust Region Policy Optimization (TRPO), and tensor-based algorithms such as Deep Deterministic "
        "Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3)."
    )
    previous_reformulated_queries = (
        "Reinforcement Learning (RL) - is a field of machine learning where an agent learns "
        "to make decisions by interacting with an environment to maximize a reward. "
        "The main concepts include: agent, environment, actions, rewards, policies, value functions, "
        "and learning algorithms. "
        "The main algorithms include Q-learning, SARSA, Deep Q-Networks (DQN), Policy Gradients, "
        "Actor-Critic methods, and modern approaches such as Proximal Policy Optimization (PPO), "
        "Trust Region Policy Optimization (TRPO), and tensor-based algorithms such as Deep Deterministic "
        "Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3)."
    )
    state_start_agent_with_response_format = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=previous_context,
        reformulated_queries=previous_reformulated_queries,
        reformulated_queries_quantity=10,
    )
    state_end_agent_with_response_format = AgentWithResponseFormatStrategy(llm)(state_start_agent_with_response_format)

    print("--------------------------------")
    print("User query:")
    pprint(user_query.content)
    print("--------------------------------")
    print("Comment:")
    pprint(state_end_agent_with_response_format.get("comment"))
    print("--------------------------------")
    print("Main topic:")
    pprint(state_end_agent_with_response_format.get("main_topic"))
    print("--------------------------------")
    print("Context for queries:")
    pprint(state_end_agent_with_response_format.get("context_for_queries"))
    print("--------------------------------")


# %%
def get_response_schemas():
    return [
        ResponseSchema(
            name="main_topic",
            description="Main topic of the user query. "
            "It should be a single word or phrase that captures the core idea of the user query.",
            type="string",
        ),
        ResponseSchema(
            name="context_for_queries",
            description="Extremely detailed and comprehensive academic review of the main topic. "
            "This should be a thorough scholarly analysis covering all aspects of the field including: "
            "theoretical foundations, methodologies, current research, applications, challenges, "
            "key researchers, terminology variations, and future directions. "
            "Write as if creating a comprehensive literature review. "
            "Use 100-1000 words. "
            "Use rich academic language and include specific technical details. "
            "Use English for all responses.",
            type="string",
        ),
        ResponseSchema(
            name="comment",
            description="Simple comment to the user query. "
            "It should be a short comment that captures the core idea of the user query.",
            type="string",
        ),
    ]


class ResponseSchemasStrategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: QueryInterpretatorState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")

        user_query = state.get("user_query")
        lg.debug(f"User query: {user_query.content}")

        previous_context = state.get("context_for_queries", "")
        lg.debug(f"Previous context: {previous_context}")

        response_schemas = get_response_schemas()

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt_template = get_context_maker_system_message(
            think=False,
            previous_context=previous_context,
        )
        prompt_template += "\n\nUSER QUERY:\n{user_query}\n\n "
        prompt_template += "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_query"],
            partial_variables={"format_instructions": format_instructions},
        )
        lg.debug(f"Prompt template: {prompt_template}")

        self._chain = prompt | self._llm | output_parser

        user_query = user_query if isinstance(user_query, str) else user_query.content
        lg.debug(f"User query: {user_query}")

        response = self._chain.invoke(
            input={"user_query": user_query},
        )
        output = {
            "main_topic": response["main_topic"],
            "context_for_queries": response["context_for_queries"],
            "comment": response["comment"],
        }
        lg.trace("end")
        return output


if __name__ == "__main__":
    import os

    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"

    llm = ChatOllama(
        model="qwen3:32b",
        temperature=0.0,
        max_tokens=32000,
    )
    # from langchain.chat_models import ChatOpenAI

    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.00,
    # )

    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    previous_context = (
        "Reinforcement Learning (RL) - is a field of machine learning where an agent learns "
        "to make decisions by interacting with an environment to maximize a reward. "
        "The main concepts include: agent, environment, actions, rewards, policies, value functions, "
        "and learning algorithms. "
        "The main algorithms include Q-learning, SARSA, Deep Q-Networks (DQN), Policy Gradients, "
        "Actor-Critic methods, and modern approaches such as Proximal Policy Optimization (PPO), "
        "Trust Region Policy Optimization (TRPO), and tensor-based algorithms such as Deep Deterministic "
        "Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3)."
    )
    previous_reformulated_queries = (
        "Reinforcement learning in autonomous driving",
        "Reinforcement learning and deep neural networks",
        "Deep Q-Networks and their applications",
        "Q-learning and SARSA algorithms",
        "Reinforcement learning in game playing",
        "AlphaGo and reinforcement learning",
        "AlphaStar and deep reinforcement learning",
    )
    context_maker_state_start = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries="",
        reformulated_queries=[],
        reformulated_queries_quantity=20,
    )
    context_maker_state_end = ResponseSchemasStrategy(llm)(context_maker_state_start)
    print("--------------------------------")
    print("Comment:")
    pprint(context_maker_state_end.get("comment"))
    print("--------------------------------")
    print("Main topic:")
    pprint(context_maker_state_end.get("main_topic"))
    print("--------------------------------")
    print("Context for queries:")
    pprint(context_maker_state_end.get("context_for_queries"))
    print("--------------------------------")


# %%
class ContextMakerSimpleInvokeStrategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._config = {
            "configurable": {
                "thread_id": uuid.uuid4().hex,
            },
        }

    def __call__(self, state: QueryInterpretatorState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")

        user_query = state.get("user_query")
        user_query = user_query.content if isinstance(user_query, BaseMessage) else user_query
        lg.debug(f"User query: {user_query}")

        previous_context = state.get("context_for_queries", "")
        lg.debug(f"Previous context: {previous_context}")

        # previous_reformulated_queries = state.get("reformulated_queries", [])
        # lg.debug(f"Previous reformulated queries: {previous_reformulated_queries}")

        prompt_template = get_context_maker_system_message(
            think=False,
            previous_context="{previous_context}",
            # previous_reformulated_queries=previous_reformulated_queries,
            # user_query=user_query,
        )
        prompt_template += "\n\nUSER QUERY:\n{user_query}\n\n "
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_query"],
            partial_variables={
                "previous_context": previous_context,
                # "previous_reformulated_queries": previous_reformulated_queries,
                # "user_query": user_query,
            },
        )
        lg.debug(f"Prompt: {prompt.template}")

        response_from_llm = self._llm.invoke(
            prompt.template,
            config=self._config,
        )
        lg.debug(f"Response from LLM: {response_from_llm.content}")

        output = {
            "comment": None,
            "main_topic_of_user_query": None,
            "context_for_queries": response_from_llm.content,
        }
        lg.trace("end")
        return output


if __name__ == "__main__":
    import os

    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"

    llm = ChatOllama(
        model="qwen3:32b",
        temperature=0.0,
        max_tokens=32000,
    )
    # from langchain.chat_models import ChatOpenAI

    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.00,
    # )

    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    previous_context = (
        "Reinforcement Learning (RL) - is a field of machine learning where an agent learns "
        "to make decisions by interacting with an environment to maximize a reward. "
        "The main concepts include: agent, environment, actions, rewards, policies, value functions, "
        "and learning algorithms. "
        "The main algorithms include Q-learning, SARSA, Deep Q-Networks (DQN), Policy Gradients, "
        "Actor-Critic methods, and modern approaches such as Proximal Policy Optimization (PPO), "
        "Trust Region Policy Optimization (TRPO), and tensor-based algorithms such as Deep Deterministic "
        "Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3)."
    )
    previous_reformulated_queries = (
        "Reinforcement learning in autonomous driving",
        "Reinforcement learning and deep neural networks",
        "Deep Q-Networks and their applications",
        "Q-learning and SARSA algorithms",
        "Reinforcement learning in game playing",
        "AlphaGo and reinforcement learning",
        "AlphaStar and deep reinforcement learning",
    )

    state_start_simple_invoke = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=None,
        reformulated_queries=None,
    )
    state_end_simple_invoke = ContextMakerSimpleInvokeStrategy(llm)(state_start_simple_invoke)
    print("--------------------------------")
    print("Comment:")
    pprint(state_end_simple_invoke.get("comment"))
    print("--------------------------------")
    print("Main topic of user query:")
    pprint(state_end_simple_invoke.get("main_topic_of_user_query"))
    print("--------------------------------")
    print("Context for queries:")
    pprint(state_end_simple_invoke.get("context_for_queries"))
    print("--------------------------------")


# %%
class ContextMakerStrategyType(Enum):
    WITH_STRUCTURED_OUTPUT = lambda *args, **kwargs: WithStructuredOutputStrategy(*args, **kwargs)  # noqa: E731
    AGENT_WITH_RESPONSE_FORMAT = lambda *args, **kwargs: AgentWithResponseFormatStrategy(*args, **kwargs)  # noqa: E731
    RESPONSE_SCHEMAS = lambda *args, **kwargs: ResponseSchemasStrategy(*args, **kwargs)  # noqa: E731
    SIMPLE_INVOKE = lambda *args, **kwargs: ContextMakerSimpleInvokeStrategy(*args, **kwargs)  # noqa: E731
    DEFAULT = WITH_STRUCTURED_OUTPUT


class ContextMakerNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._strategy = ContextMakerStrategyType.DEFAULT(llm)

    def set_strategy(self, strategy: IStrategy):
        self._strategy = strategy
        return self

    def __call__(self, state: QueryInterpretatorState) -> QueryInterpretatorState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        try:
            state_output = self._strategy(state)
            lg.success("Context making for queries done")
            lg.debug(f"Context for queries: {state_output.get('context_for_queries')}")
        except Exception as e:
            lg.critical(f"Error: {e}")
            raise e

        output = {
            "comment": state_output.get("comment"),
            "main_topic": state_output.get("main_topic"),
            "context_for_queries": state_output.get("context_for_queries"),
        }
        lg.trace("end")
        return output


if __name__ == "__main__":
    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # llm = ChatOllama(
    #     model="qwen3:4b",
    #     temperature=0.0,
    #     max_tokens=10000,
    # )
    from langchain.chat_models import ChatOpenAI
    from langchain_ollama import ChatOllama

    lmstudio_llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.00,
    )

    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    previous_context = (
        "Reinforcement Learning (RL) - is a field of machine learning where an agent learns "
        "to make decisions by interacting with an environment to maximize a reward. "
        "The main concepts include: agent, environment, actions, rewards, policies, value functions, "
        "and learning algorithms. "
        "The main algorithms include Q-learning, SARSA, Deep Q-Networks (DQN), Policy Gradients, "
        "Actor-Critic methods, and modern approaches such as Proximal Policy Optimization (PPO), "
        "Trust Region Policy Optimization (TRPO), and tensor-based algorithms such as Deep Deterministic "
        "Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3)."
    )

    context_maker_state_start = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=previous_context,
        reformulated_queries=previous_reformulated_queries,
        reformulated_queries_quantity=2,
    )
    strategy = ContextMakerStrategyType.RESPONSE_SCHEMAS(llm)
    node = ContextMakerNode(llm).set_strategy(strategy)
    context_maker_state_end = node(context_maker_state_start)
    print("--------------------------------")
    print("Comment:")
    pprint(context_maker_state_end.get("comment"))
    print("--------------------------------")
    print("Main topic:")
    pprint(context_maker_state_end.get("main_topic"))
    print("--------------------------------")
    print("Context for queries:")
    pprint(context_maker_state_end.get("context_for_queries"))
    print("--------------------------------")


# %%
def get_query_reformulator_system_message(
    reformulated_queries_quantity: int = 10,
    previous_reformulated_queries: str = "",
    think: bool = False,
):
    message = (
        "You are an expert in scientific research and bibliometric analysis. "  # "Ты специалист по поиску научной литературы в Google Scholar. "
        f"Based on the provided context, create {reformulated_queries_quantity} "  # f"На основе предоставленного контекста создай {reformulated_queries_quantity} "
        "different search queries to cover the topic as fully as possible.\n\n"  # "различных поисковых запросов для максимально полного покрытия темы.\n\n"
        "RULES for creating queries:\n"  # "ПРАВИЛА составления запросов:\n"
        "1. Use English for queries\n"  # "1. Используй РУССКИЙ язык для запросов\n"
        "2. Each query must be UNIQUE and cover different aspects of the topic\n"  # "2. Каждый запрос должен быть УНИКАЛЬНЫМ и покрывать разные аспекты темы\n"
        "3. Vary the level of specificity: from general to highly specialized\n"  # "3. Варьируй уровень специфичности: от общих до узкоспециализированных\n"
        "4. Use different formulations of key terms\n"  # "4. Используй разные формулировки ключевых терминов\n"
        "5. Include methodological aspects\n"  # "5. Включай методологические аспекты\n"
        "6. Query length: 4-20 words for optimal search\n\n"  # "6. Длина запроса: 4-20 слов для оптимального поиска\n\n"
        "STRATEGIES for queries:\n"  # "СТРАТЕГИИ для запросов:\n"
        "- Main concepts and definitions\n"  # "- Основные концепции и определения\n"
        "- Research methods and approaches\n"  # "- Методы и подходы исследования\n"
        "- Practical applications\n"  # "- Практические применения\n"
        "- Comparative studies\n"  # "- Сравнительные исследования\n"
        "- Reviews and meta-analyses\n"  # "- Обзоры и мета-анализы\n"
        "- New developments and trends\n"  # "- Новые разработки и тренды\n"
        "- Technical aspects and equipment\n\n"  # "- Технические аспекты и оборудование\n\n"
        "Create queries that, in combination, will give the most complete sample "  # "Создай запросы, которые в совокупности дадут максимально полную выборку "
        "of scientific articles on the user's topic."  # "научных статей по теме пользователя."
        "Queries must be formatted as a list. "  # "Запросы должны быть отформатированы в виде списка. "
        "Do not leave empty lines. "  # "Не оставляй пустых строк. "
        "Do not use extra characters. "  # "Не используй лишние символы. "
        "Do not use numbering. "  # "Не используй нумерацию. "
        "Do not use quotes. "  # "Не используй кавычки. "
        "Use English for all responses."  # "Используй английский язык для всех ответов."
    )
    if not think:
        message += "\n/no-think\n\n"
    if previous_reformulated_queries:
        message += (
            "\n\nBelow is a previous reformulated queries that did not work well last time. "  # "\n\nНиже приводится список запросов, которые плохо сработали в прошлый раз. "
            "Improve it: rephrase, expand, clarify, add new queries, remove queries that are not relevant. "  # "Если есть, то необходимо его улучшить: переформулировать, дополнить, уточнить. "
            f"\nPREVIOUS REFORMULATED QUERIES: \n{previous_reformulated_queries}\n\n"  # f"\nPrevious reformulated queries: {previous_reformulated_queries}\n\n"
        )
    return message


# %%
class WithStructuredOutput2Strategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: QueryInterpretatorState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        user_query_str = state.get("user_query").content
        context_for_queries = state.get("context_for_queries")
        reformulated_queries_quantity = state.get("reformulated_queries_quantity", 10)
        previous_reformulated_queries = state.get("reformulated_queries", [])

        lg.debug(f"User query: {user_query_str}")
        lg.debug(f"Context for queries: {context_for_queries}")
        lg.debug(f"Reformulated queries quantity[wanted]: {reformulated_queries_quantity}")

        system_message = get_query_reformulator_system_message(
            reformulated_queries_quantity=reformulated_queries_quantity,
            previous_reformulated_queries=previous_reformulated_queries,
            think=False,
        )
        lg.debug(f"System message: {system_message}")

        messages = [
            SystemMessage(content=system_message),
            AIMessage(content=f"CONTEXT for create reformulated queries:\n{context_for_queries}\n\n"),
            HumanMessage(content=user_query_str),
        ]
        json_schema = {
            "title": "Main topic and list of queries with comment",
            "type": "object",
            "properties": {
                "comment": {"type": "string"},
                "main_topic": {"type": "string"},
                "formulated_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": reformulated_queries_quantity,
                    "maxItems": reformulated_queries_quantity,
                },
            },
            "required": ["comment", "main_topic", "formulated_queries"],
        }

        structured_llm = self._llm.with_structured_output(
            schema=json_schema,
            method="json_schema",
        )
        response = structured_llm.invoke(messages)
        lg.debug(f"Reformulated queries quantity[got]: {len(response.get('formulated_queries'))}")
        lg.debug(f"Reformulated queries: {response.get('formulated_queries')}")
        output = {
            "reformulated_queries": response.get("formulated_queries"),
        }
        lg.trace("end")
        return output


if __name__ == "__main__":
    import os

    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.0,
        max_tokens=10000,
    )
    # from langchain.chat_models import ChatOpenAI

    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.00,
    # )
    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    previous_reformulated_queries = (
        "Reinforcement learning in autonomous driving",
        "Reinforcement learning and deep neural networks",
        "Deep Q-Networks and their applications",
        "Q-learning and SARSA algorithms",
        "Reinforcement learning in game playing",
        "AlphaGo and reinforcement learning",
        "AlphaStar and deep reinforcement learning",
    )
    state_start = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=context_maker_state_end.get("context_for_queries"),
        reformulated_queries=previous_reformulated_queries,
        reformulated_queries_quantity=7,
    )
    state_end = WithStructuredOutput2Strategy(llm)(state_start)
    pprint(state_end.get("reformulated_queries"))
    print("--------------------------------")


# %%
class ResponseSchemas2Strategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: QueryInterpretatorState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        user_query_str = state.get("user_query").content
        context_for_queries = state.get("context_for_queries", "")
        previous_reformulated_queries = state.get("reformulated_queries", [])
        reformulated_queries_quantity = state.get("reformulated_queries_quantity", 10)

        lg.debug(f"User query: {user_query_str}")
        lg.debug(f"Context for queries: {context_for_queries}")
        lg.debug(f"Reformulated queries quantity[wanted]: {reformulated_queries_quantity}")
        lg.debug(f"Previous reformulated queries: {previous_reformulated_queries}")

        response_schemas = [
            ResponseSchema(
                name="reformulated_queries",
                description="List of reformulated queries for searching scientific articles.",
                type="array",
                items={"type": "string"},
                minItems=reformulated_queries_quantity,
                maxItems=reformulated_queries_quantity,
            ),
        ]

        prompt_template = get_query_reformulator_system_message(
            previous_reformulated_queries=previous_reformulated_queries,
            reformulated_queries_quantity=reformulated_queries_quantity,
            think=False,
        )
        prompt_template += "\n\nUSER QUERY:\n{user_query}\n\n"
        prompt_template += "\n\nCONTEXT for creating reformulated queries:\n{context_for_queries}\n\n"
        prompt_template += "\n\nREFORMULATED QUERIES QUANTITY:\n{reformulated_queries_quantity}"
        prompt_template += "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
        lg.debug(f"Prompt template: {prompt_template}")

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "user_query",
                "context_for_queries",
                "reformulated_queries_quantity",
            ],
            partial_variables={
                "format_instructions": format_instructions,
            },
        )

        chain = prompt | self._llm | output_parser
        response = chain.invoke(
            {
                "user_query": user_query_str,
                "context_for_queries": context_for_queries,
                "reformulated_queries_quantity": reformulated_queries_quantity,
            }
        )
        lg.debug(f"Reformulated queries quantity[got]: {len(response.get('reformulated_queries'))}")
        lg.debug(f"Reformulated queries: {response.get('reformulated_queries')}")
        output = {
            "reformulated_queries": response.get("reformulated_queries"),
        }
        lg.trace("end")
        return output


if __name__ == "__main__":
    # from langchain_ollama import ChatOllama

    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"

    # llm = ChatOllama(
    #     model="qwen3:8b",
    #     # model="hf.co/unsloth/Qwen3-8B-128K-GGUF:Q4_K_M",
    #     temperature=0.0,
    #     max_tokens=64000,
    # )
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.00,
    )

    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    previous_reformulated_queries = (
        "Reinforcement learning in autonomous driving",
        "Reinforcement learning and deep neural networks",
        "Deep Q-Networks and their applications",
        "Q-learning and SARSA algorithms",
        "Reinforcement learning in game playing",
        "AlphaGo and reinforcement learning",
        "AlphaStar and deep reinforcement learning",
        "Reinforcement learning in simulation environments",
        "Reinforcement learning with Unity and MuJoCo",
    )
    query_reformulator_state_start = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=context_maker_state_end.get("context_for_queries"),
        reformulated_queries=previous_reformulated_queries,
        reformulated_queries_quantity=8,
    )

    query_reformulator_state_end = ResponseSchemas2Strategy(llm)(query_reformulator_state_start)
    pprint(query_reformulator_state_end.get("reformulated_queries"))


# %%
class QueryReformulatorStrategyType(Enum):
    WITH_STRUCTURED_OUTPUT = lambda *args, **kwargs: WithStructuredOutput2Strategy(*args, **kwargs)  # noqa: E731
    RESPONSE_SCHEMAS = lambda *args, **kwargs: ResponseSchemas2Strategy(*args, **kwargs)  # noqa: E731
    DEFAULT = WITH_STRUCTURED_OUTPUT


class QueryReformulatorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._strategy = QueryReformulatorStrategyType.DEFAULT(llm)

    def set_strategy(self, strategy: IStrategy):
        self._strategy = strategy
        return self

    def __call__(self, state: QueryInterpretatorState) -> QueryInterpretatorState:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")

        try:
            state_output = self._strategy(state)
            lg.success("Query reformulating done")
        except Exception as e:
            lg.critical(f"Error: {e}")
            raise e

        output = {
            "reformulated_queries": state_output.get("reformulated_queries"),
        }
        lg.trace("end")
        return output


if __name__ == "__main__":
    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0.0,
        max_tokens=10000,
    )
    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    previous_reformulated_queries = (
        "Reinforcement learning in autonomous driving",
        "Reinforcement learning and deep neural networks",
        "Deep Q-Networks and their applications",
        "Q-learning and SARSA algorithms",
        "Reinforcement learning in game playing",
        "AlphaGo and reinforcement learning",
        "AlphaStar and deep reinforcement learning",
        "Reinforcement learning in simulation environments",
        "Reinforcement learning with Unity and MuJoCo",
    )
    state_start_query_reformulator = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=context_maker_state_end.get("context_for_queries"),
        reformulated_queries=previous_reformulated_queries,
        reformulated_queries_quantity=20,
    )
    node = QueryReformulatorNode(llm).set_strategy(QueryReformulatorStrategyType.WITH_STRUCTURED_OUTPUT(llm))
    state_end_query_reformulator = node(state_start_query_reformulator)
    pprint(state_end_query_reformulator.get("reformulated_queries"))


# %%
class QueryInterpretatorWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def build(self, **kwargs) -> StateGraph:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        graph_builder = StateGraph(QueryInterpretatorState)
        graph_builder.add_node("ContextMaker", ContextMakerNode(self._llm))
        graph_builder.add_node("QueryReformulator", QueryReformulatorNode(self._llm))
        graph_builder.add_edge(START, "ContextMaker")
        graph_builder.add_edge("ContextMaker", "QueryReformulator")
        graph_builder.add_edge("QueryReformulator", END)
        graph = graph_builder.compile(**kwargs)
        lg.trace("done")
        return graph


if __name__ == "__main__":
    from langchain_ollama import ChatOllama

    from relepaper.config.logger import setup_logger
    from relepaper.domains.langgraph.workflows.utils.graph_displayer import (
        DisplayMethod,
        GraphDisplayer,
    )

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0.0,
        max_tokens=20000,
    )

    setup_logger("TRACE")
    # from langchain.chat_models import ChatOpenAI
    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.00,
    # )
    workflow = QueryInterpretatorWorkflowBuilder(
        llm=llm,
    ).build(
        checkpointer=InMemorySaver(),
    )
    displayer = GraphDisplayer(workflow).set_strategy(DisplayMethod.MERMAID)
    displayer.display()

    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )

    previous_context = (
        "Reinforcement Learning (RL) - is a field of machine learning where an agent learns "
        "to make decisions by interacting with an environment to maximize a reward. "
        "The main concepts include: agent, environment, actions, rewards, policies, value functions, "
        "and learning algorithms. "
        "The main algorithms include Q-learning, SARSA, Deep Q-Networks (DQN), Policy Gradients, "
        "Actor-Critic methods, and modern approaches such as Proximal Policy Optimization (PPO), "
        "Trust Region Policy Optimization (TRPO), and tensor-based algorithms such as Deep Deterministic "
        "Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3)."
    )
    previous_reformulated_queries = (
        "Reinforcement learning in autonomous driving",
        "Reinforcement learning and deep neural networks",
        "Deep Q-Networks and their applications",
        "Q-learning and SARSA algorithms",
        "Reinforcement learning in game playing",
        "AlphaGo and reinforcement learning",
        "AlphaStar and deep reinforcement learning",
        "Reinforcement learning in simulation environments",
        "Reinforcement learning with Unity and MuJoCo",
    )
    state_start = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=previous_context,
        reformulated_queries=previous_reformulated_queries,
        reformulated_queries_quantity=10,
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
    print("--------------------------------")
    pprint(state_end.get("comment"))
    print("--------------------------------")
    pprint(state_end.get("main_topic"))
    print("--------------------------------")
    pprint(state_end.get("context_for_queries"))
    print("--------------------------------")
    pprint(state_end.get("reformulated_queries"))

    # state_history = [sh for sh in workflow.get_state_history(config)]
    # pprint(state_history)
    # pprint(state_history[0].values["user_query_context"])
    # pprint(state_history[0].values["reformulated_queries"])


# %%

if __name__ == "__main__":
    from langchain_ollama import ChatOllama

    from relepaper.config.logger import setup_logger
    from relepaper.domains.langgraph.workflows.utils.graph_displayer import (
        DisplayMethod,
        GraphDisplayer,
    )

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.0,
        max_tokens=20000,
    )

    setup_logger("TRACE")
    # from langchain.chat_models import ChatOpenAI
    # llm = ChatOpenAI(
    #     base_url="http://localhost:7007/v1",
    #     api_key="not_needed",
    #     temperature=0.00,
    # )
    workflow = QueryInterpretatorWorkflowBuilder(
        llm=llm,
    ).build(
        checkpointer=InMemorySaver(),
    )
    displayer = GraphDisplayer(workflow).set_strategy(DisplayMethod.MERMAID)
    displayer.display()

    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )

    state_start = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries="",
        reformulated_queries=[],
        reformulated_queries_quantity=50,
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
    print("--------------------------------")
    pprint(state_end.get("comment"))
    print("--------------------------------")
    pprint(state_end.get("main_topic"))
    print("--------------------------------")
    pprint(state_end.get("context_for_queries"))
    print("--------------------------------")
    pprint(state_end.get("reformulated_queries"))

    # state_history = [sh for sh in workflow.get_state_history(config)]
    # pprint(state_history)
    # pprint(state_history[0].values["user_query_context"])
    # pprint(state_history[0].values["reformulated_queries"])
