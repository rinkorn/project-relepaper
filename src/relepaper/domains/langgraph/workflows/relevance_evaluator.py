# %%
import logging
import random
from pprint import pprint
from typing import List, TypedDict

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from relepaper.domains.langgraph.workflows.interfaces import (
    IWorkflowBuilder,
    IWorkflowNode,
)
from relepaper.domains.langgraph.workflows.pdf_analyser import (
    extract_metadata_from_pdf,
    read_pdf_content,
)
from relepaper.domains.langgraph.workflows.utils import display_graph
from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.openalex.entities.work import OpenAlexWork

__all__ = [
    "RelevanceEvaluatorState",
    "RelevanceEvaluatorWorkflowBuilder",
]

# %%
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    formatter = logging.Formatter("%(message)s")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# %%
class RelevanceEvaluatorState(TypedDict):
    user_query: BaseMessage
    works: List[OpenAlexWork]
    pdfs: List[OpenAlexPDF]
    pdfs_metadatas: List[dict]
    pdfs_contents: List[str]
    pdfs_scores: List[float]


# %%
class PDFContentExtractorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._config = {
            "configurable": {
                "use_async": False,
                "use_multithreading": False,
                "batch_size": 2,
                "max_concurrency": 2,
                "show_progress": True,
            }
        }

    def __call__(self, state: RelevanceEvaluatorState) -> RelevanceEvaluatorState:
        logger.info(":::CALL_PDF_CONTENT_EXTRACTOR:::")

        pdf_contents = []
        pdf_metadatas = []
        for pdf in state["pdfs"]:
            pdf_content, pdf_metadata = read_pdf_content(pdf.dirname / pdf.filename, num_pages=-1)
            extracted_metadata = extract_metadata_from_pdf(pdf_content, pdf_metadata, self._llm)
            pdf_contents.append(pdf_content)
            pdf_metadatas.append(extracted_metadata)

        output = {
            "pdfs": state["pdfs"],
            "pdfs_contents": pdf_contents,
            "pdfs_metadatas": pdf_metadatas,
        }

        return output


if __name__ == "__main__":
    from pathlib import PosixPath

    # import os
    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # llm = ChatOllama(
    #     # model="qwen3:30B-a3b",
    #     # model="hf.co/unsloth/Qwen3-30B-A3B-128K-GGUF:Q4_1",
    #     model="hf.co/unsloth/Qwen3-14B-128K-GGUF:Q4_K_M",
    #     # model="qwen3:32b",
    #     # model="qwen3:14b",
    #     # model="qwen3:8b",
    #     temperature=0.0,
    #     max_tokens=128000,
    # )
    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.0,
    )
    pdfs = [
        OpenAlexPDF(
            url="https://www.mdpi.com/1996-1073/10/11/1846/pdf?version=1510484667",
            dirname=PosixPath(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="energies-10-01846.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-021-00561-9",
            dirname=PosixPath(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="s13321-021-00561-9.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=PosixPath(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="Intl J Robust   Nonlinear - 2021 - Wan - Optimal control and learning for cyber‐physical systems.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
    ]

    pdf_content_state_start = RelevanceEvaluatorState(
        pdfs=pdfs,
    )
    node = PDFContentExtractorNode(llm=llm)
    pdf_content_state_end = node(pdf_content_state_start)
    pprint(pdf_content_state_end["pdfs_contents"])
    pprint(pdf_content_state_end["pdfs_metadatas"])


# %%
class PDFRelevanceEvaluatorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._config = {
            "configurable": {
                "use_async": False,
                "use_multithreading": False,
                "batch_size": 2,
                "max_concurrency": 2,
                "show_progress": True,
            }
        }

    def __call__(self, state: RelevanceEvaluatorState) -> RelevanceEvaluatorState:
        logger.info(":::CALL_PDF_CONTENT_ANALYZER:::")

        user_query = state["user_query"]

        response_schemas = [
            ResponseSchema(
                name="theme_score",
                description="Score for the thematic correspondence. The score is a number between 0 and 100.",
                type="number",
                minValue=0,
                maxValue=100,
                multipleOf=1,
            ),
            ResponseSchema(
                name="terminology_score",
                description="Score for the terminology correspondence. The score is a number between 0 and 100.",
                type="number",
                minValue=0,
                maxValue=100,
                multipleOf=1,
            ),
            ResponseSchema(
                name="methodology_score",
                description="Score for the methodology correspondence. The score is a number between 0 and 100.",
                type="number",
                minValue=0,
                maxValue=100,
                multipleOf=1,
            ),
            ResponseSchema(
                name="practical_applicability_score",
                description="Score for the practical applicability. The score is a number between 0 and 100.",
                type="number",
                minValue=0,
                maxValue=100,
                multipleOf=1,
            ),
            ResponseSchema(
                name="novelty_and_relevance_score",
                description="Score for the novelty and relevance. The score is a number between 0 and 100.",
                type="number",
                minValue=0,
                maxValue=100,
                multipleOf=1,
            ),
            ResponseSchema(
                name="fundamental_significance_score",
                description="Score for the fundamental significance. The score is a number between 0 and 100.",
                type="number",
                minValue=0,
                maxValue=100,
                multipleOf=1,
            ),
            ResponseSchema(
                name="overall_score",
                description=(
                    "Overall score for the relevance of the article. The score is a number between 0 and 100. "
                    "The score is the mean of the scores of the following criteria: theme_score, terminology_score, methodology_score, practical_applicability_score, novelty_and_relevance_score, fundamental_significance_score"
                ),
                type="number",
                minValue=0,
                maxValue=100,
                multipleOf=1,
            ),
        ]

        prompt_template = (
            "Ты эксперт по оценке релевантности научных статей. "
            "Твоя задача - оценить, насколько найденная статья соответствует запросу пользователя.\n\n"
            "КРИТЕРИИ ОЦЕНКИ:\n"
            "1. ДОМЕННОЕ СООТВЕТСТВИЕ (0-100 баллов):\n"
            "   - Прямое совпадение доменов: 75-100 баллов\n"
            "   - Частичное совпадение доменов: 50-74 балла\n"
            "   - Косвенная связь с доменами: 25-49 баллов\n"
            "   - Нет соответствия доменам: 0-24 балла\n\n"
            "2. ТЕМАТИЧЕСКОЕ СООТВЕТСТВИЕ (0-100 баллов):\n"
            "   - Прямое совпадение основной темы: 75-100 баллов\n"
            "   - Частичное совпадение темы: 50-74 балла\n"
            "   - Косвенная связь с темой: 25-49 баллов\n"
            "   - Нет тематической связи: 0-24 балла\n\n"
            "3. ТЕРМИНОЛОГИЧЕСКОЕ СОВПАДЕНИЕ (0-100 баллов):\n"
            "   - Все ключевые термины присутствуют: 75-100 баллов\n"
            "   - Большинство терминов совпадает: 50-74 баллов\n"
            "   - Частичное совпадение терминов: 25-49 баллов\n"
            "   - Нет совпадающих терминов: 0-24 баллов\n\n"
            "4. МЕТОДОЛОГИЧЕСКАЯ РЕЛЕВАНТНОСТЬ (0-100 баллов):\n"
            "   - Методы полностью соответствуют: 75-100 баллов\n"
            "   - Схожие методы исследования: 25-49 баллов\n"
            "   - Частично применимые методы: 1-24 баллов\n"
            "   - Разные методологические подходы: 0-24 баллов\n\n"
            "5. ПРАКТИЧЕСКАЯ ПРИМЕНИМОСТЬ (0-100 баллов):\n"
            "   - Прямо применимо к теме исследования: 75-100 баллов\n"
            "   - Косвенно применимо: 25-49 баллов\n"
            "   - Ограниченная применимость: 1-24 баллов\n"
            "   - Не применимо: 0-24 баллов\n\n"
            "6. НОВИЗНА И АКТУАЛЬНОСТЬ (0-100 баллов):\n"
            "   - Свежие исследования (последние 2-3 года): 75-100 баллов\n"
            "   - Относительно новые (3-5 лет): 50-74 баллов\n"
            "   - Частично новые (6-10 лет): 25-49 баллов\n"
            "   - Устаревшие исследования (более 10 лет): 0-24 баллов\n\n"
            "7. ФУНДАМЕНТАЛЬНАЯ ЗНАЧИМОСТЬ (0-100 баллов):\n"
            "   - Высокая фундаментальная значимость: 75-100 баллов\n"
            "   - Средняя фундаментальная значимость: 50-74 баллов\n"
            "   - Частичная фундаментальная значимость: 25-49 баллов\n"
            "   - Низкая фундаментальная значимость: 0-24 баллов\n\n"
            "ОБЩАЯ ОЦЕНКА:\n"
            "- 80-100 баллов: ВЫСОКАЯ релевантность (обязательно включить)\n"
            "- 50-79 балла: СРЕДНЯЯ релевантность (рекомендуется включить)\n"
            "- 20-49 балла: НИЗКАЯ релевантность (рассмотреть включение)\n"
            "- 0-19 баллов: НЕ РЕЛЕВАНТНО (исключить)\n\n"
            "ФОРМАТ ОТВЕТА:\n"
            "Оценка по критериям:\n"
            "- Тематическое соответствие: X/100\n"
            "- Терминологическое совпадение: X/100\n"
            "- Методологическая релевантность: X/100\n"
            "- Практическая применимость: X/100\n"
            "- Новизна и актуальность: X/100\n"
            "- Фундаментальная значимость: X/100\n"
            "ИТОГО: X/100\n\n"
            "Уровень релевантности: [ВЫСОКАЯ/СРЕДНЯЯ/НИЗКАЯ/НЕ РЕЛЕВАНТНО]\n\n"
            "Обоснование:\n"
            "[Краткое объяснение оценки с указанием сильных и слабых сторон соответствия]\n\n"
            "ИСХОДНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_query}\n\n"
            "ИНФОРМАЦИЯ О СТАТЬЕ (источник: {journal}):\n"
            "TITLE: {title}\n\n"
            "ABSTRACT: {abstract}\n\n"
            "KEYWORDS: {keywords}\n\n"
            "YEAR: {year}\n\n"
            # "PDF CONTENT: {pdf_content}\n\n"
            "FORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
        )
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_query", "title", "abstract", "keywords", "journal", "year", "pdf_content"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | llm | output_parser

        responses = chain.batch(
            inputs=[
                {
                    "user_query": user_query,
                    "title": pdf_metadata["title"],
                    "abstract": pdf_metadata["abstract"],
                    "keywords": pdf_metadata["keywords"],
                    "journal": pdf_metadata["journal"],
                    "year": pdf_metadata["year"],
                    "pdf_content": pdf_content,
                }
                for pdf_metadata, pdf_content in zip(state["pdfs_metadatas"], state["pdfs_contents"])
            ],
            config=self._config,
        )
        output = {
            "pdfs_scores": [response["overall_score"] for response in responses],
            "responses": responses,
        }
        return output


if __name__ == "__main__":
    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # llm = ChatOllama(
    #     # model="qwen3:30B-a3b",
    #     # model="hf.co/unsloth/Qwen3-30B-A3B-128K-GGUF:Q4_1",
    #     model="hf.co/unsloth/Qwen3-14B-128K-GGUF:Q4_K_M",
    #     # model="qwen3:32b",
    #     # model="qwen3:14b",
    #     # model="qwen3:8b",
    #     temperature=0.0,
    #     max_tokens=128000,
    # )
    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.0,
    )

    user_queries = [
        "Поведение собак в террариуме",
        "Строительство домов из бетона",
        "Фотосинтез в растениях. Выращивание растений в гараже",
        "Машинное обучение. Компьютерное зрение",
        "Машинное обучение. Обучение с подкреплением",
        "Менеджмент энергии в микросетке с использованием пакетного обучения с подкреплением",
        "Battery Energy Management in a Microgrid Using Batch Reinforcement Learning",
    ]
    for user_query in user_queries:
        print("|" * 100)
        evaluator_state_start = RelevanceEvaluatorState(
            user_query=HumanMessage(
                content=(f"Я пишу диссертацию по теме: {user_query}. Скачай все статьи по этой теме. \n/no-think"),
            ),
            pdfs=pdf_content_state_end["pdfs"],
            pdfs_metadatas=pdf_content_state_end["pdfs_metadatas"],
            pdfs_contents=pdf_content_state_end["pdfs_contents"],
            pdfs_scores=[],
        )

        node = PDFRelevanceEvaluatorNode(llm=llm)
        evaluator_state_end = node(evaluator_state_start)
        pprint(evaluator_state_start["user_query"].content)
        for i, pdf_metadata in enumerate(evaluator_state_start["pdfs_metadatas"]):
            pprint(pdf_metadata["title"])
            # pprint(pdf_metadata["abstract"])
            pprint(pdf_metadata["keywords"])
            pprint(pdf_metadata["journal"])
            pprint(pdf_metadata["year"])
            pprint(evaluator_state_end["responses"][i])
            print("-" * 100)

        pprint(evaluator_state_end["pdfs_scores"])


# %%
class RelevanceEvaluatorWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _relevance_evaluator_node(self, state: RelevanceEvaluatorState) -> dict:
        logger.info(":::CALL_RELEVANCE_EVALUATOR:::")
        pdfs = state["pdfs"]
        messages = state["messages"]

        scores: List[float] = []
        for pdf in pdfs:
            score = round(random.random() * 10, 2)
            scores.append(score)

        return {
            "messages": messages,
            "pdfs_scores": scores,
        }

    def build(self, **kwargs) -> StateGraph:
        graph_builder = StateGraph(RelevanceEvaluatorState)
        graph_builder.add_node("RelevanceEvaluator", self._relevance_evaluator_node)
        graph_builder.add_edge(START, "RelevanceEvaluator")
        graph_builder.add_edge("RelevanceEvaluator", END)
        compiled_graph = graph_builder.compile()
        return compiled_graph


# %%
if __name__ == "__main__":
    from relepaper.config.dev_settings import get_dev_settings

    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.0,
        max_tokens=10000,
    )

    openalex_pdfs = [
        OpenAlexPDF(
            url="http://www.jbc.org/article/S0021925820496280/pdf",
            dirname=get_dev_settings().project_path / "data" / "pdf",
            filename="PIIS0021925820496280.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://www.chinesechemsoc.org/doi/pdf/10.31635/ccschem.020.202000271",
            dirname=get_dev_settings().project_path / "data" / "pdf",
            filename="wang-et-al-degradation-mechanisms-in-blue-organic-light-emitting-diodes.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
    ]

    workflow = RelevanceEvaluatorWorkflowBuilder(llm=llm).build(checkpointer=InMemorySaver())
    display_graph(workflow)

    state_start = RelevanceEvaluatorState(
        pdfs=openalex_pdfs,
        pdfs_scores=[],
    )
    state_end = workflow.invoke(input=state_start)

    for pdf, score in zip(state_end["pdfs"], state_end["pdfs_scores"]):
        print(f"PDF: {pdf.filename}\n\tscore: {score}")
