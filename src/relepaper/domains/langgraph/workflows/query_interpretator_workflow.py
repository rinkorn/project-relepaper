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
def get_context_maker_system_message(think: bool = True):
    return (
        "You are an expert in scientific research and bibliometric analysis. "  # "Ты эксперт по научным исследованиям и библиометрическому анализу. "
        "Your task is to analyze the user query and create a detailed context "  # "Твоя задача - проанализировать пользовательский запрос, выделить главную тему и создать детальный контекст "
        "for finding relevant scientific articles.\n\n"  # "для поиска релевантных научных статей в Google Scholar.\n\n"
        "ANALYZE the user query and DETERMINE:\n"  # "АНАЛИЗИРУЙ запрос пользователя и ОПРЕДЕЛИ:\n"
        "1. Main topic of the user query. "  # "1. Главную тему исследования\n"
        "2. Main scientific field and discipline\n"  # "2. Основную научную область и дисциплину\n"
        "3. Key terms and concepts (including synonyms and English equivalents)\n"  # "3. Ключевые термины и понятия (включая синонимы и англоязычные эквиваленты)\n"
        "4. Research methods that may be relevant\n"  # "4. Методы исследования, которые могут быть релевантны\n"
        "5. Related areas of knowledge\n"  # "5. Смежные области знаний\n"
        "6. Temporal context of research (if specified)\n"  # "6. Временной контекст исследований (если указан)\n"
        "7. Level of research (theoretical, experimental, applied)\n\n"  # "7. Уровень исследования (теоретический, экспериментальный, прикладной)\n\n"
        "CREATE a structured context, including:\n"  # "СОЗДАЙ структурированный контекст, включающий:\n"
        "- Alternative formulations of terms\n"  # "- Альтернативные формулировки терминов\n"
        "- Related concepts and areas\n"  # "- Смежные понятия и области\n"
        "- Potential methodological approaches\n"  # "- Потенциальные методологические подходы\n"
        "- Specific technologies or equipment (if mentioned)\n\n"  # "- Специфические технологии или оборудование (если упомянуты)\n\n"
        "Use English for all responses.\n"  # "Используй РУССКИЙ язык для всех ответов.\n"
        "The context should be detailed enough to create a variety of search queries, "  # "Контекст должен быть достаточно подробным для создания разнообразных поисковых запросов, "
        "but focused on the main topic of the research."  # "но сфокусированным на основной теме исследования. "
        "\n/no-think"
        if not think
        else ""  # "\n/no-think"
    )


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
def get_structured_output_schema():
    return {
        "title": "Queries main topic and context with comment",
        "type": "object",
        "properties": {
            "comment": {
                "type": "string",
                "description": "Comment to the user query. "
                "It should be a short comment that captures the core idea of the user query.",
            },
            "main_topic": {
                "type": "string",
                "description": "Main topic of the user query. "
                "It should be a single word or phrase that captures the core idea of the user query.",
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
        self._system_message = get_context_maker_system_message(think=False)
        self._json_schema = get_structured_output_schema()
        self._config = {
            "configurable": {
                "thread_id": uuid.uuid4().hex,
            },
        }

    def __call__(self, state: QueryInterpretatorState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.info("start")

        messages = [
            SystemMessage(content=self._system_message),
            HumanMessage(content=state.get("user_query").content),
        ]
        structured_llm = self._llm.with_structured_output(
            schema=self._json_schema,
            method="json_schema",
        )
        response = structured_llm.invoke(messages, config=self._config)
        output = {
            "comment": response["comment"],
            "main_topic": response["main_topic"],
            "context_for_queries": response["context_for_queries"],
        }
        lg.info("end")
        return output


if __name__ == "__main__":
    import os

    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        # model="qwen3:32b",
        model="qwen3:8b",
        temperature=0.0,
        max_tokens=10000,
    )
    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    state_start_with_structured_output = QueryInterpretatorState(
        user_query=user_query,
        reformulated_queries_quantity=10,
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
        self._system_message = get_context_maker_system_message(think=False)

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
            prompt=self._system_message,
            name="context_maker",
            response_format=AnswerResponseFormat,
            # checkpointer=InMemorySaver(),
        )
        self._config = {
            "configurable": {
                "thread_id": uuid.uuid4().hex,
            },
        }

    def __call__(self, state: QueryInterpretatorState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.info("start")

        user_query = state.get("user_query")
        user_query = user_query if isinstance(user_query, HumanMessage) else HumanMessage(content=user_query)
        lg.debug(f"User query: {user_query.content}")

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
        # model="qwen3:32b",
        model="qwen3:8b",
        temperature=0.0,
        max_tokens=10000,
    )

    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    state_start_agent_with_response_format = QueryInterpretatorState(
        user_query=user_query,
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
        response_schemas = get_response_schemas()

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt_template = get_context_maker_system_message(think=False)
        prompt_template += "\n\nUSER QUERY:\n{user_query}\n\n "
        prompt_template += "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_query"],
            partial_variables={"format_instructions": format_instructions},
        )
        self._chain = prompt | self._llm | output_parser

    def __call__(self, state: QueryInterpretatorState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        user_query = state.get("user_query")
        lg.debug(f"User query: {user_query}")
        user_query = user_query if isinstance(user_query, str) else user_query.content

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
    # from langchain_ollama import ChatOllama
    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"

    # llm = ChatOllama(
    #     # model="qwen3:30B-a3b",
    #     model="qwen3:32b",
    #     # model="qwen3:14b",
    #     # model="qwen3:8b",
    #     temperature=0.0,
    #     max_tokens=10000,
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
    context_maker_state_start = QueryInterpretatorState(
        user_query=user_query,
        reformulated_queries_quantity=10,
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
class ContextMakerStrategyType(Enum):
    WITH_STRUCTURED_OUTPUT = lambda *args, **kwargs: WithStructuredOutputStrategy(*args, **kwargs)  # noqa: E731
    AGENT_WITH_RESPONSE_FORMAT = lambda *args, **kwargs: AgentWithResponseFormatStrategy(*args, **kwargs)  # noqa: E731
    RESPONSE_SCHEMAS = lambda *args, **kwargs: ResponseSchemasStrategy(*args, **kwargs)  # noqa: E731


class ContextMakerNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._strategy = ContextMakerStrategyType.RESPONSE_SCHEMAS(llm)

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
    from langchain_ollama import ChatOllama

    os.environ["OLLAMA_HOST"] = "http://localhost:11434"

    ollama_llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.0,
        max_tokens=10000,
    )
    from langchain.chat_models import ChatOpenAI

    lmstudio_llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.00,
    )

    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )

    context_maker_state_start = QueryInterpretatorState(
        user_query=user_query,
        reformulated_queries_quantity=10,
    )
    strategy = ContextMakerStrategyType.RESPONSE_SCHEMAS(ollama_llm)
    node = ContextMakerNode(ollama_llm).set_strategy(strategy)
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
def get_query_reformulator_system_message(reformulated_queries_quantity: int, think: bool = True):
    return (
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
        f"\n/no-think"
        if not think
        else ""
    )


class WithStructuredOutput2Strategy(IStrategy):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: QueryInterpretatorState) -> dict:
        lg = logger.bind(classname=self.__class__.__name__)
        lg.trace("start")
        user_query_str = state.get("user_query").content
        context_for_queries = state.get("context_for_queries")
        reformulated_queries_quantity = state.get("reformulated_queries_quantity", 10)

        lg.debug(f"User query: {user_query_str}")
        lg.debug(f"Context for queries: {context_for_queries}")
        lg.debug(f"Reformulated queries quantity[wanted]: {reformulated_queries_quantity}")

        system_message = get_query_reformulator_system_message(
            reformulated_queries_quantity=reformulated_queries_quantity,
            think=False,
        )

        messages = [
            SystemMessage(content=system_message),
            AIMessage(content=f"CONTEXT for creating queries:\n{context_for_queries}\n\n"),
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
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.0,
        max_tokens=10000,
    )
    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    state_start = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=context_maker_state_end.get("context_for_queries"),
        reformulated_queries_quantity=20,
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
        context_for_queries = state.get("context_for_queries")
        reformulated_queries_quantity = state.get("reformulated_queries_quantity", 10)

        lg.debug(f"User query: {user_query_str}")
        lg.debug(f"Context for queries: {context_for_queries}")
        lg.debug(f"Reformulated queries quantity[wanted]: {reformulated_queries_quantity}")

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
            reformulated_queries_quantity=reformulated_queries_quantity,
            think=False,
        )
        prompt_template += "\n\nUSER QUERY:\n{user_query}\n\n"
        prompt_template += "\n\nCONTEXT for creating queries:\n{context_for_queries}\n\n"
        prompt_template += "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "user_query",
                "context_for_queries",
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
    query_reformulator_state_start = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=context_maker_state_end.get("context_for_queries"),
        reformulated_queries_quantity=20,
    )

    query_reformulator_state_end = ResponseSchemas2Strategy(llm)(query_reformulator_state_start)
    pprint(query_reformulator_state_end.get("reformulated_queries"))


# %%
class QueryReformulatorStrategyType(Enum):
    WITH_STRUCTURED_OUTPUT = lambda *args, **kwargs: WithStructuredOutput2Strategy(*args, **kwargs)  # noqa: E731
    RESPONSE_SCHEMAS = lambda *args, **kwargs: ResponseSchemas2Strategy(*args, **kwargs)  # noqa: E731


class QueryReformulatorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._strategy = QueryReformulatorStrategyType.RESPONSE_SCHEMAS(llm)

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
        model="qwen3:8b",
        temperature=0.0,
        max_tokens=10000,
    )
    user_query = HumanMessage(
        content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
        "Скачай все статьи по этой теме. /no-think",
    )
    state_start_query_reformulator = QueryInterpretatorState(
        user_query=user_query,
        context_for_queries=context_maker_state_end.get("context_for_queries"),
        reformulated_queries_quantity=20,
    )
    node = QueryReformulatorNode(llm).set_strategy(QueryReformulatorStrategyType.RESPONSE_SCHEMAS(llm))
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
    from relepaper.domains.langgraph.workflows.utils.graph_displayer import (
        DisplayMethod,
        GraphDisplayer,
    )
    from relepaper.config.logger import setup_logger

    from langchain_ollama import ChatOllama

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


# %%

if __name__ == "__main__":
    from relepaper.domains.langgraph.workflows.utils.graph_displayer import (
        DisplayMethod,
        GraphDisplayer,
    )
    from relepaper.config.logger import setup_logger

    from langchain_ollama import ChatOllama

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
