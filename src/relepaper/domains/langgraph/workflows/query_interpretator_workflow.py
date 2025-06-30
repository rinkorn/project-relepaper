# %%
import os
import uuid
from pprint import pprint
from typing import List, TypedDict

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from relepaper.domains.langgraph.entities.session import Session
from relepaper.domains.langgraph.workflows.interfaces import IWorkflowBuilder, IWorkflowNode
from relepaper.domains.langgraph.workflows.utils import display_graph

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
    comment: str
    reformulated_queries_quantity: int = 10
    reformulated_queries: List[str]


# %%
# class ContextMakerNode(IWorkflowNode):
#     def __init__(self, llm: BaseChatModel):
#         self._llm = llm

#     def __call__(self, state: QueryInterpretatorState) -> QueryInterpretatorState:
#         logger.info(":::NODE: ContextMaker:::")
#         user_query_str = state.get("user_query").content

#         # system_message = (
#         #     "Ты эксперт по научным исследованиям и библиометрическому анализу. "
#         #     "Твоя задача - проанализировать пользовательский запрос, выделить главную тему и создать детальный контекст "
#         #     "для поиска релевантных научных статей в Google Scholar.\n\n"
#         #     "АНАЛИЗИРУЙ запрос пользователя и ОПРЕДЕЛИ:\n"
#         #     "1. Главную тему исследования\n"
#         #     "2. Основную научную область и дисциплину\n"
#         #     "3. Ключевые термины и понятия (включая синонимы и англоязычные эквиваленты)\n"
#         #     "4. Методы исследования, которые могут быть релевантны\n"
#         #     "5. Смежные области знаний\n"
#         #     "6. Временной контекст исследований (если указан)\n"
#         #     "7. Уровень исследования (теоретический, экспериментальный, прикладной)\n\n"
#         #     "СОЗДАЙ структурированный контекст, включающий:\n"
#         #     "- Альтернативные формулировки терминов\n"
#         #     "- Смежные понятия и области\n"
#         #     "- Потенциальные методологические подходы\n"
#         #     "- Специфические технологии или оборудование (если упомянуты)\n\n"
#         #     "Используй РУССКИЙ язык для всех ответов.\n"
#         #     "Контекст должен быть достаточно подробным для создания разнообразных поисковых запросов, "
#         #     "но сфокусированным на основной теме исследования. "
#         #     "Рассуждай последовательно и детально. "
#         #     "Не замыкайся. "
#         #     "\n/no-think"
#         # )

#         system_message = (
#             "You are an expert in scientific research and bibliometric analysis. "
#             "Your task is to analyze the user query and create a detailed context "
#             "for finding relevant scientific articles.\n\n"
#             "ANALYZE the user query and DETERMINE:\n"
#             "1. Main topic of the user query. "
#             "2. Main scientific field and discipline\n"
#             "3. Key terms and concepts (including synonyms and English equivalents)\n"
#             "4. Research methods that may be relevant\n"
#             "5. Related areas of knowledge\n"
#             "6. Temporal context of research (if specified)\n"
#             "7. Level of research (theoretical, experimental, applied)\n\n"
#             "CREATE a structured context, including:\n"
#             "- Alternative formulations of terms\n"
#             "- Related concepts and areas\n"
#             "- Potential methodological approaches\n"
#             "- Specific technologies or equipment (if mentioned)\n\n"
#             "Use English for all responses.\n"
#             "The context should be detailed enough to create a variety of search queries, "
#             "but focused on the main topic of the research."
#             "\n/no-think"
#         )

#         messages = [
#             SystemMessage(content=system_message),
#             HumanMessage(content=user_query_str),
#         ]
#         json_schema = {
#             "title": "Queries main topic and context with comment",
#             "type": "object",
#             "properties": {
#                 "comment": {
#                     "type": "string",
#                     "description": "Comment to the user query. "
#                     "It should be a short comment that captures the core idea of the user query.",
#                 },
#                 "main_topic": {
#                     "type": "string",
#                     "description": "Main topic of the user query. "
#                     "It should be a single word or phrase that captures the core idea of the user query.",
#                 },
#                 "context_for_queries": {
#                     "type": "string",
#                     "description": "Context for creating queries. "
#                     "It should be a detailed description of the user query and the main topic of the user query.",
#                 },
#             },
#             "required": ["comment", "main_topic", "context_for_queries"],
#         }
#         structured_llm = self._llm.with_structured_output(
#             schema=json_schema,
#             method="json_schema",
#         )
#         llm_output = structured_llm.invoke(messages)
#         comment = llm_output["comment"]
#         main_topic = llm_output["main_topic"]
#         context_for_queries = llm_output["context_for_queries"]
#         output = {
#             "comment": comment,
#             "main_topic": main_topic,
#             "context_for_queries": context_for_queries,
#         }
#         return output


# if __name__ == "__main__":
#     from langchain.chat_models import ChatOpenAI

#     llm = ChatOpenAI(
#         base_url="http://localhost:7007/v1",
#         api_key="not_needed",
#         temperature=0.00,
#     )
#     # llm = ChatOllama(
#     #     # model="qwen3:30B-a3b",
#     #     model="qwen3:32b",
#     #     # model="qwen3:14b",
#     #     # model="qwen3:8b",
#     #     temperature=0.0,
#     #     max_tokens=10000,
#     # )
#     context_maker_state_start = QueryInterpretatorState(
#         user_query=HumanMessage(
#             content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
#             "Скачай все статьи по этой теме. /no-think",
#         ),
#         reformulated_queries_quantity=10,
#     )
#     context_maker_state_end = ContextMakerNode(llm)(context_maker_state_start)
#     print("--------------------------------")
#     print("Comment:")
#     pprint(context_maker_state_end.get("comment"))
#     print("--------------------------------")
#     print("Main topic:")
#     pprint(context_maker_state_end.get("main_topic"))
#     print("--------------------------------")
#     print("Context for queries:")
#     pprint(context_maker_state_end.get("context_for_queries"))
#     print("--------------------------------")


# %%
# class ContextMaker2Node(IWorkflowNode):
#     def __init__(self, llm: BaseChatModel):
#         self._llm = llm
#         # self._system_message = (
#         #     "Ты эксперт по научным исследованиям и библиометрическому анализу. "
#         #     "Твоя задача - проанализировать пользовательский запрос, выделить главную тему и создать детальный контекст "
#         #     "для поиска релевантных научных статей в Google Scholar.\n\n"
#         #     "АНАЛИЗИРУЙ запрос пользователя и ОПРЕДЕЛИ:\n"
#         #     "1. Главную тему исследования\n"
#         #     "2. Основную научную область и дисциплину\n"
#         #     "3. Ключевые термины и понятия (включая синонимы и англоязычные эквиваленты)\n"
#         #     "4. Методы исследования, которые могут быть релевантны\n"
#         #     "5. Смежные области знаний\n"
#         #     "6. Временной контекст исследований (если указан)\n"
#         #     "7. Уровень исследования (теоретический, экспериментальный, прикладной)\n\n"
#         #     "СОЗДАЙ структурированный контекст, включающий:\n"
#         #     "- Альтернативные формулировки терминов\n"
#         #     "- Смежные понятия и области\n"
#         #     "- Потенциальные методологические подходы\n"
#         #     "- Специфические технологии или оборудование (если упомянуты)\n\n"
#         #     "Используй РУССКИЙ язык для всех ответов.\n"
#         #     "Контекст должен быть достаточно подробным для создания разнообразных поисковых запросов, "
#         #     "но сфокусированным на основной теме исследования. "
#         #     "Рассуждай последовательно и детально. "
#         #     "\n/no-think"
#         # )
#         self._system_message = (
#             "You are an expert in scientific research and bibliometric analysis. "
#             "Your task is to analyze the user query and create a detailed context "
#             "for finding relevant scientific articles.\n\n"
#             "ANALYZE the user query and DETERMINE:\n"
#             "1. Main scientific field and discipline\n"
#             "2. Key terms and concepts (including synonyms and English equivalents)\n"
#             "3. Research methods that may be relevant\n"
#             "4. Related areas of knowledge\n"
#             "5. Temporal context of research (if specified)\n"
#             "6. Level of research (theoretical, experimental, applied)\n\n"
#             "CREATE a structured context, including:\n"
#             "- Alternative formulations of terms\n"
#             "- Related concepts and areas\n"
#             "- Potential methodological approaches\n"
#             "- Specific technologies or equipment (if mentioned)\n\n"
#             "Use English for all responses.\n"
#             "The context should be detailed enough to create a variety of search queries, "
#             "but focused on the main topic of the research."
#             "\n/no-think"
#         )

#         from pydantic import BaseModel, Field
#         from langgraph.prebuilt import create_react_agent

#         class AnswerResponseFormat(BaseModel):
#             comment: str = Field(description="Comment to the user query")
#             main_topic: str = Field(description="Main topic of the user query")
#             query_analysis: str = Field(description="Analysis of the user query")
#             context_for_queries: str = Field(description="Context for creating queries")

#         self._agent = create_react_agent(
#             model=llm,
#             tools=[],
#             prompt=self._system_message,
#             name="context_maker2",
#             response_format=AnswerResponseFormat,
#             # checkpointer=InMemorySaver(),
#         )
#         self._config = {
#             "configurable": {
#                 "thread_id": uuid.uuid4().hex,
#             },
#         }

#     def __call__(self, state: QueryInterpretatorState) -> QueryInterpretatorState:
#         logger.info(":::NODE: ContextMaker:::")
#         user_query = state.get("user_query")

#         if isinstance(user_query, HumanMessage):
#             user_query = user_query
#         else:
#             user_query = HumanMessage(content=user_query)

#         agent_output = self._agent.invoke(
#             input=(user_query),
#             config=self._config,
#         )
#         output = {
#             "comment": agent_output["structured_response"].__dict__["comment"],
#             "query_analysis": agent_output["structured_response"].__dict__["query_analysis"],
#             "main_topic": agent_output["structured_response"].__dict__["main_topic"],
#             "context_for_queries": agent_output["structured_response"].__dict__["context_for_queries"],
#             "agent_output": agent_output["messages"][0].content,
#         }
#         pprint(agent_output["messages"][0].content)
#         # pprint(output)
#         return output


# if __name__ == "__main__":
#     from langchain.chat_models import ChatOpenAI
#     llm = ChatOpenAI(
#         base_url="http://localhost:7007/v1",
#         api_key="not_needed",
#         temperature=0.00,
#     )
#     # llm = ChatOllama(
#     #     # model="qwen3:30B-a3b",
#     #     model="qwen3:32b",
#     #     # model="qwen3:14b",
#     #     # model="qwen3:8b",
#     #     temperature=0.0,
#     #     max_tokens=10000,
#     # )
#     state_start = QueryInterpretatorState(
#         user_query=HumanMessage(
#             content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
#             "Скачай все статьи по этой теме. /no-think",
#         ),
#         reformulated_queries_quantity=10,
#     )
#     state_end = ContextMaker2Node(llm)(state_start)
#     # pprint(state_end.get("comment"))
#     # pprint(state_end.get("main_topic"))
#     # print("--------------------------------")
#     # print("Messages:")
#     # pprint(state_end.get("agent_output"))
#     print("--------------------------------")
#     print("Comment:")
#     pprint(state_end.get("comment"))
#     print("--------------------------------")
#     print("User query analysis:")
#     pprint(state_end.get("query_analysis"))
#     print("--------------------------------")
#     print("Main topic:")
#     pprint(state_end.get("main_topic"))
#     print("--------------------------------")
#     print("Context for queries:")
#     pprint(state_end.get("context_for_queries"))


# %%
class ContextMaker3Node(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        response_schemas = [
            ResponseSchema(
                name="main_topic",
                description="Main topic of the user query. "
                "It should be a single word or phrase that captures the core idea of the user query.",
                type="string",
            ),
            ResponseSchema(
                name="context_for_queries",
                description="Fully detailed context for creating queries. "
                "It should be a detailed description of the user query and the main topic of the user query.",
                type="string",
            ),
            ResponseSchema(
                name="comment",
                description="Simple comment to the user query. "
                "It should be a short comment that captures the core idea of the user query.",
                type="string",
            ),
        ]
        # prompt_template = (
        #     "Ты эксперт по научным исследованиям и библиометрическому анализу. "
        #     "Твоя задача - проанализировать пользовательский запрос, выделить главную тему и создать детальный контекст "
        #     "для поиска релевантных научных статей в Google Scholar.\n\n"
        #     "АНАЛИЗИРУЙ запрос пользователя и ОПРЕДЕЛИ:\n"
        #     "1. Главную тему исследования\n"
        #     "2. Основную научную область и дисциплину\n"
        #     "3. Ключевые термины и понятия (включая синонимы и англоязычные эквиваленты)\n"
        #     "4. Методы исследования, которые могут быть релевантны\n"
        #     "5. Смежные области знаний\n"
        #     "6. Временной контекст исследований (если указан)\n"
        #     "7. Уровень исследования (теоретический, экспериментальный, прикладной)\n\n"
        #     "СОЗДАЙ структурированный контекст, включающий:\n"
        #     "- Альтернативные формулировки терминов\n"
        #     "- Смежные понятия и области\n"
        #     "- Потенциальные методологические подходы\n"
        #     "- Специфические технологии или оборудование (если упомянуты)\n\n"
        #     "Используй РУССКИЙ язык для всех ответов.\n"
        #     "Контекст должен быть достаточно подробным для создания разнообразных поисковых запросов, "
        #     "но сфокусированным на основной теме исследования. "
        #     "Рассуждай последовательно и детально. "
        #     "\n\nUSER QUERY:\n{user_query}\n\n "
        #     "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
        #     "\n/no-think"
        # )
        prompt_template = (
            "You are an expert in scientific research and bibliometric analysis. "
            "Your task is to analyze the user query and create a detailed context "
            "for finding relevant scientific articles.\n\n"
            "ANALYZE the user query and DETERMINE:\n"
            "1. Main topic of the user query. It should be a single word or phrase that captures the core idea of the user query.\n"
            "2. Main scientific field and discipline\n"
            "3. Key terms and concepts (including synonyms and English equivalents)\n"
            "4. Research methods that may be relevant\n"
            "5. Related areas of knowledge\n"
            "6. Temporal context of research (if specified)\n"
            "7. Level of research (theoretical, experimental, applied)\n"
            "CREATE a structured context, including:\n"
            "- Alternative formulations of terms\n"
            "- Related concepts and areas\n"
            "- Potential methodological approaches\n"
            "- Specific technologies or equipment (if mentioned)\n\n"
            "Use English for all responses.\n"
            "The context should be detailed enough to create a variety of search queries, "
            "but focused on the main topic of the research."
            "Think step by step and in detail. "
            "\n\nUSER QUERY:\n{user_query}\n\n "
            "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
            "\n/no-think"
        )
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_query"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | self._llm | output_parser
        self._chain = chain

    def __call__(self, state: QueryInterpretatorState) -> QueryInterpretatorState:
        logger.info(":::NODE: ReformulateQuery:::")
        user_query = state.get("user_query")
        if isinstance(user_query, str):
            user_query = HumanMessage(content=user_query)
        elif isinstance(user_query, BaseMessage):
            user_query = user_query
        else:
            raise ValueError(f"Invalid user query type: {type(user_query)}.")

        response = self._chain.invoke(
            {
                "user_query": user_query,
            }
        )
        output = {
            "main_topic": response["main_topic"],
            "context_for_queries": response["context_for_queries"],
            "comment": response["comment"],
        }
        return output


if __name__ == "__main__":
    llm = ChatOllama(
        # model="qwen3:30B-a3b",
        model="qwen3:32b",
        # model="qwen3:14b",
        # model="qwen3:8b",
        temperature=0.0,
        max_tokens=10000,
    )
    context_maker_state_start = QueryInterpretatorState(
        user_query=HumanMessage(
            content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
            "Скачай все статьи по этой теме. /no-think",
        ),
        reformulated_queries_quantity=10,
    )
    context_maker_state_end = ContextMaker3Node(llm)(context_maker_state_start)
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
# class QueryReformulatorNode(IWorkflowNode):
#     def __init__(self, llm: BaseChatModel):
#         self._llm = llm

#     def __call__(self, state: QueryInterpretatorState) -> dict:
#         logger.info(":::NODE: QueryReformulator:::")
#         user_query_str = state.get("user_query").content
#         context_for_queries = state.get("context_for_queries")
#         reformulated_queries_quantity = state.get("reformulated_queries_quantity", 10)

#         # system_message = (
#         #     f"Ты специалист по поиску научной литературы в Google Scholar. "
#         #     f"На основе предоставленного контекста создай {reformulated_queries_quantity} "
#         #     "различных поисковых запросов для максимально полного покрытия темы.\n\n"
#         #     "ПРАВИЛА составления запросов:\n"
#         #     "1. Используй РУССКИЙ язык для запросов\n"
#         #     "2. Каждый запрос должен быть УНИКАЛЬНЫМ и покрывать разные аспекты темы\n"
#         #     "3. Варьируй уровень специфичности: от общих до узкоспециализированных\n"
#         #     "4. Используй разные формулировки ключевых терминов\n"
#         #     "5. Включай методологические аспекты\n"
#         #     "6. Длина запроса: 4-12 слов для оптимального поиска\n\n"
#         #     "СТРАТЕГИИ для запросов:\n"
#         #     "- Основные концепции и определения\n"
#         #     "- Методы и подходы исследования\n"
#         #     "- Практические применения\n"
#         #     "- Сравнительные исследования\n"
#         #     "- Обзоры и мета-анализы\n"
#         #     "- Новые разработки и тренды\n"
#         #     "- Технические аспекты и оборудование\n\n"
#         #     "Создай запросы, которые в совокупности дадут максимально полную выборку "
#         #     "научных статей по теме пользователя."
#         #     "Запросы должны быть отформатированы в виде списка. "
#         #     "Не оставляй пустых строк. "
#         #     "Не используй лишние символы. "
#         #     "Не используй нумерацию. "
#         #     "\n/no-think"
#         # )
#         system_message = (
#             f"You are an expert in scientific research and bibliometric analysis. "
#             f"Based on the provided context, create {reformulated_queries_quantity} "
#             "different search queries to cover the topic as fully as possible.\n\n"
#             "RULES for creating queries:\n"
#             "1. Use English for queries\n"
#             "2. Each query must be UNIQUE and cover different aspects of the topic\n"
#             "3. Vary the level of specificity: from general to highly specialized\n"
#             "4. Use different formulations of key terms\n"
#             "5. Include methodological aspects\n"
#             "6. Query length: 4-20 words for optimal search\n\n"
#             "STRATEGIES for queries:\n"
#             "- Main concepts and definitions\n"
#             "- Research methods and approaches\n"
#             "- Practical applications\n"
#             "- Comparative studies\n"
#             "- Reviews and meta-analyses\n"
#             "- New developments and trends\n"
#             "- Technical aspects and equipment\n\n"
#             "Create queries that, in combination, will give the most complete sample "
#             "of scientific articles on the user's topic."
#             "Queries must be formatted as a list. "
#             "Do not leave empty lines. "
#             "Do not use extra characters. "
#             "Do not use numbering. "
#             "Do not use quotes. "
#             "\n/no-think"
#         )

#         messages = [
#             SystemMessage(content=system_message),
#             AIMessage(content=f"CONTEXT for creating queries:\n{context_for_queries}\n\n"),
#             HumanMessage(content=user_query_str),
#         ]
#         json_schema = {
#             "title": "Main topic and list of queries with comment",
#             "type": "object",
#             "properties": {
#                 "comment": {"type": "string"},
#                 "main_topic": {"type": "string"},
#                 "formulated_queries": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "minItems": reformulated_queries_quantity,
#                     "maxItems": reformulated_queries_quantity,
#                 },
#             },
#             "required": ["comment", "main_topic", "formulated_queries"],
#         }

#         structured_llm = self._llm.with_structured_output(
#             schema=json_schema,
#             method="json_schema",
#         )
#         structured_output = structured_llm.invoke(messages)
#         comment = structured_output["comment"]
#         main_topic = structured_output["main_topic"]
#         formulated_queries = structured_output["formulated_queries"]

#         output = {
#             "comment": comment,
#             "main_topic": main_topic,
#             "reformulated_queries": formulated_queries,
#         }
#         return output


# # if __name__ == "__main__":
# #     llm = ChatOllama(
# #         model="qwen3:30B-a3b",
# #         temperature=0.0,
# #         max_tokens=10000,
# #     )
# #     state_start = QueryInterpretatorState(
# #         user_query=HumanMessage(
# #             content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
# #             "Скачай все статьи по этой теме. /no-think",
# #         ),
# #     )
# #     state_end = QueryReformulatorNode(llm)(state_start)
# #     pprint(state_end.get("comment"))
# #     pprint(state_end.get("main_topic"))
# #     pprint(state_end.get("reformulated_queries"))


# %%
class QueryReformulator2Node(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: QueryInterpretatorState) -> dict:
        logger.info(":::NODE: QueryReformulator:::")
        user_query_str = state.get("user_query").content
        context_for_queries = state.get("context_for_queries")
        reformulated_queries_quantity = state.get("reformulated_queries_quantity", 10)

        response_schemas = [
            ResponseSchema(
                name="reformulated_queries",
                description="List of reformulated queries for searching scientific articles",
                type="array",
                items={"type": "string"},
                minItems=reformulated_queries_quantity,
                maxItems=reformulated_queries_quantity,
            ),
        ]

        # prompt_template = (
        #     "Ты специалист по поиску научной литературы в Google Scholar. "
        #     "На основе предоставленного контекста создай {reformulated_queries_quantity} "
        #     "различных поисковых запросов для максимально полного покрытия темы.\n\n"
        #     "ПРАВИЛА составления запросов:\n"
        #     "1. Используй РУССКИЙ язык для запросов\n"
        #     "2. Каждый запрос должен быть УНИКАЛЬНЫМ и покрывать разные аспекты темы\n"
        #     "3. Варьируй уровень специфичности: от общих до узкоспециализированных\n"
        #     "4. Используй разные формулировки ключевых терминов\n"
        #     "5. Включай методологические аспекты\n"
        #     "6. Длина запроса: 4-12 слов для оптимального поиска\n\n"
        #     "СТРАТЕГИИ для запросов:\n"
        #     "- Основные концепции и определения\n"
        #     "- Методы и подходы исследования\n"
        #     "- Практические применения\n"
        #     "- Сравнительные исследования\n"
        #     "- Обзоры и мета-анализы\n"
        #     "- Новые разработки и тренды\n"
        #     "- Технические аспекты и оборудование\n\n"
        #     "Создай запросы на АНГЛИЙСКОМ языке, которые в совокупности дадут максимально полную "
        #     "выборку научных статей по теме пользователя."
        #     "Запросы должны быть отформатированы в виде списка. "
        #     "Не оставляй пустых строк. "
        #     "Не используй лишние символы. "
        #     "Не используй нумерацию. "
        #     "\n\nUSER QUERY:\n{user_query}\n\n"
        #     "\n\nCONTEXT for creating queries:\n{context_for_queries}\n\n"
        #     "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
        #     # "\n\nREFORMULATED QUERIES QUANTITY:\n{reformulated_queries_quantity}\n\n"
        #     "\n/no-think"
        # )
        prompt_template = (
            "You are an expert in scientific research and bibliometric analysis. "
            "Based on the provided context, create {reformulated_queries_quantity} "
            "different search queries to cover the topic as fully as possible.\n\n"
            "RULES for creating queries:\n"
            "1. Use English for queries\n"
            "2. Each query must be UNIQUE and cover different aspects of the topic\n"
            "3. Vary the level of specificity: from general to highly specialized\n"
            "4. Use different formulations of key terms\n"
            "5. Include methodological aspects\n"
            "6. Query length: 4-20 words for optimal search\n\n"
            "STRATEGIES for queries:\n"
            "- Main concepts and definitions\n"
            "- Research methods and approaches\n"
            "- Practical applications\n"
            "- Comparative studies\n"
            "- Reviews and meta-analyses\n"
            "- New developments and trends\n"
            "- Technical aspects and equipment\n\n"
            "Create queries in English, which, in combination, will give the most complete sample "
            "of scientific articles on the user's topic."
            "Queries must be formatted as a list. "
            "Do not leave empty lines. "
            "Do not use extra characters. "
            "Do not use numbering. "
            "\n\nUSER QUERY:\n{user_query}\n\n"
            "\n\nCONTEXT for creating queries:\n{context_for_queries}\n\n"
            "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
            # "\n\nREFORMULATED QUERIES QUANTITY:\n{reformulated_queries_quantity}\n\n"
            "\n/no-think"
        )
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
        logger.info(prompt)
        chain = prompt | self._llm | output_parser
        response = chain.invoke(
            {
                "user_query": user_query_str,
                "context_for_queries": context_for_queries,
                "reformulated_queries_quantity": reformulated_queries_quantity,
            }
        )
        logger.info(response)
        output = {
            "reformulated_queries": response["reformulated_queries"],
        }
        return output


if __name__ == "__main__":
    llm = ChatOllama(
        # model="qwen3:30B-a3b",
        model="qwen3:32b",
        temperature=0.0,
        max_tokens=10000,
    )
    # context_maker_state_end = ContextMaker3Node(llm)(context_maker_state_start)
    query_reformulator_state_start = QueryInterpretatorState(
        user_query=HumanMessage(
            content="Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
            "Скачай все статьи по этой теме. /no-think",
        ),
        context_for_queries=context_maker_state_end.get("context_for_queries"),
        reformulated_queries_quantity=10,
    )
    query_reformulator_state_end = QueryReformulator2Node(llm)(query_reformulator_state_start)
    pprint(query_reformulator_state_end.get("reformulated_queries"))


# %%
class QueryInterpretatorWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def build(self, **kwargs) -> StateGraph:
        logger.trace("QueryInterpretatorWorkflowBuilder: build: start")
        graph_builder = StateGraph(QueryInterpretatorState)
        graph_builder.add_node("ContextMaker", ContextMaker3Node(self._llm))
        graph_builder.add_node("QueryReformulator", QueryReformulator2Node(self._llm))
        graph_builder.add_edge(START, "ContextMaker")
        graph_builder.add_edge("ContextMaker", "QueryReformulator")
        graph_builder.add_edge("QueryReformulator", END)
        graph = graph_builder.compile(**kwargs)
        logger.trace("QueryInterpretatorWorkflowBuilder: build: done")
        return graph


if __name__ == "__main__":
    # test use case
    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    llm = ChatOllama(
        model="qwen3:30B-a3b",
        # model="hf.co/unsloth/Qwen3-30B-A3B-128K-GGUF:Q4_1",
        # model="qwen3:32b",
        # model="qwen3:14b",
        # model="qwen3:8b",
        temperature=0.0,
        max_tokens=20000,
    )
    workflow = QueryInterpretatorWorkflowBuilder(
        llm=llm,
    ).build(
        checkpointer=InMemorySaver(),
    )
    display_graph(workflow)

    state_start = QueryInterpretatorState(
        user_query=HumanMessage(
            # content=("Я пишу диссертацию по теме: Машинное обучение. Скачай все статьи по этой теме. /no-think"),
            content=(
                "Я пишу диссертацию по теме: Машинное обучение. Обучение с подкреплением. "
                "Скачай все статьи по этой теме. "
                "\n/no-think"
            ),
        ),
        context_for_queries="",
        reformulated_queries=[],
        reformulated_queries_quantity=20,
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

    state_history = [sh for sh in workflow.get_state_history(config)]
    # pprint(state_history)
    # pprint(state_history[0].values["user_query_context"])
    # pprint(state_history[0].values["reformulated_queries"])


# %%
