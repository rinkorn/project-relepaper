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
from langgraph.graph import END, START, StateGraph

from relepaper.domains.langgraph.entities.session import Session
from relepaper.domains.langgraph.workflows.interfaces import (
    IWorkflowBuilder,
    IWorkflowNode,
)
from relepaper.domains.langgraph.workflows.pdf_analyser import (
    PDFMetadataExtractorState,
    PDFMetadataExtractorWorkflowBuilder,
)
from relepaper.domains.langgraph.workflows.utils import display_graph
from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.openalex.entities.work import OpenAlexWork
from relepaper.domains.pdf_exploring.external.adapters.factory import AdapterFactory
from relepaper.domains.pdf_exploring.services.pdf_content_service import PDFDocumentService

__all__ = [
    "RelevanceEvaluatorState",
    "RelevanceEvaluatorWorkflowBuilder",
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
def get_prompt_template() -> str:
    return (
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


def get_response_schemas() -> List[ResponseSchema]:
    return [
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


# %%
class RelevanceEvaluatorState(TypedDict):
    session: Session
    user_query: BaseMessage
    works: List[OpenAlexWork]
    pdfs: List[OpenAlexPDF]
    pdfs_extracted_metadata: List[PDFMetadataExtractorState]
    relevance_scores: List[float]


# %%
class PDFMetadataExtractorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel, max_concurrency: int = 2):
        self._llm = llm
        self._config = {
            "configurable": {
                "max_concurrency": max_concurrency,
            }
        }
        self._workflow = PDFMetadataExtractorWorkflowBuilder(llm=self._llm).build()
        self._pdf_adapter = AdapterFactory.create("pymupdf")
        self._pdf_service = PDFDocumentService(pdf_adapter=self._pdf_adapter)

    def __call__(self, state: RelevanceEvaluatorState) -> RelevanceEvaluatorState:
        logger.info(":::CALL_PDF_METADATA_EXTRACTOR:::")

        pdfs_extracted_metadata = []
        for openalex_pdf in state["pdfs"]:
            pdf_document = self._pdf_service.load_pdf_document(openalex_pdf.dirname / openalex_pdf.filename)
            state_input = {
                "pdf_document": pdf_document,
            }
            state_output = self._workflow.invoke(
                input=state_input,
                config=self._config,
            )
            pdfs_extracted_metadata.append(state_output)

        output = {
            "pdfs_extracted_metadata": pdfs_extracted_metadata,
        }
        return output


if __name__ == "__main__":
    from pathlib import Path
    from pprint import pprint

    llm = ChatOpenAI(
        base_url="http://localhost:7007/v1",
        api_key="not_needed",
        temperature=0.00,
    )
    pdfs = [
        OpenAlexPDF(
            url="https://www.mdpi.com/1996-1073/10/11/1846/pdf?version=1510484667",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="energies-10-01846.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="3219819.3220096.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="Intl J Robust   Nonlinear - 2021 - Wan - Optimal control and learning for cyber‐physical systems.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        # OpenAlexPDF(
        #     url="https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-021-00561-9",
        #     dirname=Path(
        #         "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
        #     ),
        #     filename="s13321-021-00561-9.pdf",
        #     strategy=PDFDownloadStrategy.SELENIUM,
        # ),
    ]
    user_query = "Менеджмент энергии в микросетке с использованием пакетного обучения с подкреплением"
    pdf_metadata_extractor_node = PDFMetadataExtractorNode(llm=llm)
    extractor_state_start = RelevanceEvaluatorState(
        user_query=HumanMessage(
            content=(f"Я пишу диссертацию по теме: {user_query}. Скачай все статьи по этой теме. \n/no-think"),
        ),
        pdfs=pdfs,
        pdfs_extracted_metadata=[],
        relevance_scores=[],
        works=None,
    )
    extractor_state_end = pdf_metadata_extractor_node(extractor_state_start)
    pprint(extractor_state_end)


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
        pdfs_extracted_metadata = state["pdfs_extracted_metadata"]

        response_schemas = get_response_schemas()
        prompt_template = get_prompt_template()
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_query", "title", "abstract", "keywords", "journal", "year", "pdf_content"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | self._llm | output_parser

        responses = chain.batch(
            inputs=[
                {
                    "user_query": user_query,
                    "title": metadata.get("extracted_title", ""),
                    "abstract": metadata.get("extracted_abstract", ""),
                    "keywords": metadata.get("extracted_keywords", []),
                    "journal": metadata.get("extracted_journal", ""),
                    "year": metadata.get("extracted_year", ""),
                    "pdf_content": metadata.get("pdf_document", "").text,
                }
                for metadata in pdfs_extracted_metadata
            ],
            config=self._config,
        )
        pprint(responses)
        for response in responses:
            del response["overall_score"]
        mean_relevance_scores = [sum(response.values()) / len(response.values()) for response in responses]
        output = {
            "relevance_scores": [round(score, 2) for score in mean_relevance_scores],
        }
        return output


if __name__ == "__main__":
    from pathlib import Path

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
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="energies-10-01846.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="3219819.3220096.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="Intl J Robust   Nonlinear - 2021 - Wan - Optimal control and learning for cyber‐physical systems.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        # OpenAlexPDF(
        #     url="https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-021-00561-9",
        #     dirname=Path(
        #         "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
        #     ),
        #     filename="s13321-021-00561-9.pdf",
        #     strategy=PDFDownloadStrategy.SELENIUM,
        # ),
    ]

    user_query = "Менеджмент энергии в микросетке с использованием пакетного обучения с подкреплением"

    print("|" * 100)
    evaluator_state_start = RelevanceEvaluatorState(
        user_query=HumanMessage(
            content=(f"Я пишу диссертацию по теме: {user_query}. Скачай все статьи по этой теме. \n/no-think"),
        ),
        pdfs=pdfs,
        pdfs_extracted_metadata=extractor_state_end["pdfs_extracted_metadata"],
        relevance_scores=[],
        works=None,
    )

    pdf_relevance_evaluator_node = PDFRelevanceEvaluatorNode(llm=llm)
    evaluator_state_end = pdf_relevance_evaluator_node(evaluator_state_start)
    pprint(evaluator_state_start["user_query"].content)
    for i, pdf in enumerate(evaluator_state_start["pdfs"]):
        print("-" * 100)
        pprint(pdf.filename)
        pprint(evaluator_state_end["relevance_scores"][i])


# %%
class RandomRelevanceEvaluatorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: RelevanceEvaluatorState) -> RelevanceEvaluatorState:
        logger.info(":::CALL_RANDOM_RELEVANCE_EVALUATOR:::")
        pdfs = state["pdfs"]

        scores: List[float] = []
        for pdf in pdfs:
            score = round(random.random() * 10, 2)
            scores.append(score)
        output = {
            "relevance_scores": scores,
        }
        return output


# %%
class RelevanceEvaluatorWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def build(self, **kwargs) -> StateGraph:
        graph_builder = StateGraph(RelevanceEvaluatorState)
        graph_builder.add_node("PDFMetadataExtractor", PDFMetadataExtractorNode(llm=self._llm))
        graph_builder.add_node("RelevanceEvaluator", PDFRelevanceEvaluatorNode(llm=self._llm))
        # graph_builder.add_node("RelevanceEvaluator", RandomRelevanceEvaluatorNode(llm=self._llm))
        graph_builder.add_edge(START, "PDFMetadataExtractor")
        graph_builder.add_edge("PDFMetadataExtractor", "RelevanceEvaluator")
        # graph_builder.add_edge(START, "RelevanceEvaluator")
        graph_builder.add_edge("RelevanceEvaluator", END)
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

    openalex_pdfs = [
        OpenAlexPDF(
            url="https://www.mdpi.com/1996-1073/10/11/1846/pdf?version=1510484667",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="energies-10-01846.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="3219819.3220096.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="Intl J Robust   Nonlinear - 2021 - Wan - Optimal control and learning for cyber‐physical systems.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
        # OpenAlexPDF(
        #     url="https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-021-00561-9",
        #     dirname=Path(
        #         "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
        #     ),
        #     filename="s13321-021-00561-9.pdf",
        #     strategy=PDFDownloadStrategy.SELENIUM,
        # ),
    ]
    workflow = RelevanceEvaluatorWorkflowBuilder(llm=llm).build()
    display_graph(workflow)

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
        state_start = RelevanceEvaluatorState(
            user_query=HumanMessage(
                content=(f"Я пишу диссертацию по теме: {user_query}. Скачай все статьи по этой теме. \n/no-think"),
            ),
            pdfs=openalex_pdfs,
            works=None,
            pdfs_extracted_metadata=[],
            relevance_scores=[],
        )
        state_end = workflow.invoke(input=state_start)

        for pdf, extracted_metadata, score in zip(
            state_end["pdfs"], state_end["pdfs_extracted_metadata"], state_end["relevance_scores"]
        ):
            print("-" * 100)
            print(f"User query: {user_query}")
            print(f"Title: {extracted_metadata.get('extracted_title', '')}")
            print(f"PDF: {pdf.filename}\n\tscore: {score}")
