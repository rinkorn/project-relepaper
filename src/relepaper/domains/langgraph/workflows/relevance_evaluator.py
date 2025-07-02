# %%
import random
from pprint import pprint
from typing import List, TypedDict

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from loguru import logger

from relepaper.domains.langgraph.entities.relevance_score import RelevanceCriteria, RelevanceScore, Score
from relepaper.domains.langgraph.entities.relevance_score_container import RelevanceScoreContainer
from relepaper.domains.langgraph.entities.session import Session
from relepaper.domains.langgraph.interfaces import IWorkflowBuilder, IWorkflowNode
from relepaper.domains.langgraph.workflows.pdf_analyser import (
    PDFAnalyserState,
    PDFAnalyserWorkflowBuilder,
)
from relepaper.domains.openalex.entities.pdf import OpenAlexPDF, PDFDownloadStrategy
from relepaper.domains.openalex.entities.work import OpenAlexWork
from relepaper.domains.pdf_exploring.external.adapters.factory import AdapterFactory
from relepaper.domains.pdf_exploring.services.pdf_content_service import PDFDocumentService

__all__ = [
    "RelevanceEvaluatorState",
    "RelevanceEvaluatorWorkflowBuilder",
]


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
        "ИНФОРМАЦИЯ О СТАТЬЕ:\n"
        "TITLE: {title}\n\n"
        "ABSTRACT: {abstract}\n\n"
        "KEYWORDS: {keywords}\n\n"
        "YEAR: {year}\n\n"
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
    ]


# %%
class RelevanceEvaluatorState(TypedDict):
    session: Session
    user_query: BaseMessage
    works: List[OpenAlexWork]
    pdfs: List[OpenAlexPDF]
    pdfs_metadata_extracted: List[PDFAnalyserState]
    relevance_scores: List[RelevanceScoreContainer]
    pdf_analyser_state: PDFAnalyserState


# %%
class PDFMetadataExtractorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel, max_concurrency: int = 2):
        self._llm = llm
        self._config = {
            "configurable": {
                "max_concurrency": max_concurrency,
            }
        }
        self._pdf_metadata_extractor_workflow = PDFAnalyserWorkflowBuilder(llm=self._llm).build()
        self._pdf_adapter = AdapterFactory.create("pymupdf")
        self._pdf_service = PDFDocumentService(pdf_adapter=self._pdf_adapter)

    def __call__(self, state: RelevanceEvaluatorState) -> RelevanceEvaluatorState:
        logger.trace(f"{self.__class__.__name__}: __call__: start")

        pdfs_metadata_extracted = []
        for openalex_pdf in state["pdfs"]:
            if not openalex_pdf.is_file_exist:
                continue
            pdf_path = openalex_pdf.file_path
            pdf_document = self._pdf_service.load_pdf_document(pdf_path)
            pdf_analyser_state = state["pdf_analyser_state"]
            short_long_pdf_length_threshold = pdf_analyser_state["short_long_pdf_length_threshold"]
            max_chunk_length = pdf_analyser_state["max_chunk_length"]
            max_chunks_count = pdf_analyser_state["max_chunks_count"]
            intersection_length = pdf_analyser_state["intersection_length"]
            state_input = {
                "pdf_document": pdf_document,
                "short_long_pdf_length_threshold": short_long_pdf_length_threshold,
                "max_chunk_length": max_chunk_length,
                "max_chunks_count": max_chunks_count,
                "intersection_length": intersection_length,
            }
            state_output = self._pdf_metadata_extractor_workflow.invoke(
                input=state_input,
                config=self._config,
            )
            pdf_metadata_extracted = state_output["pdf_metadata_extracted"]
            pdfs_metadata_extracted.append(pdf_metadata_extracted)

        output = {
            "pdfs_metadata_extracted": pdfs_metadata_extracted,
        }
        logger.trace(f"{self.__class__.__name__}: __call__: end")
        return output


if __name__ == "__main__":
    from pathlib import Path
    from pprint import pprint

    from langchain.chat_models import ChatOpenAI

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
            filename="",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
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
        OpenAlexPDF(
            url="https://dr.ntu.edu.sg/bitstream/10356/172831/2/main_thesis.pdf",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="main_thesis.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
            source_query="Менеджмент энергии в микросетке с использованием пакетного обучения с подкреплением",
        ),
        OpenAlexPDF(
            url="https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-021-00561-9",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="s13321-021-00561-9.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
    ]
    user_query = "Менеджмент энергии в микросетке с использованием пакетного обучения с подкреплением"
    pdf_metadata_extractor_node = PDFMetadataExtractorNode(llm=llm)
    extractor_state_start = RelevanceEvaluatorState(
        user_query=HumanMessage(
            content=(f"Я пишу диссертацию по теме: {user_query}. Скачай все статьи по этой теме. \n/no-think"),
        ),
        pdfs=pdfs,
        pdfs_metadata_extracted=[],
        relevance_scores=[],
        works=None,
        pdf_analyser_state=PDFAnalyserState(
            short_long_pdf_length_threshold=100000,
            max_chunk_length=100000,
            max_chunks_count=10,
            intersection_length=1000,
        ),
    )
    extractor_state_end = pdf_metadata_extractor_node(extractor_state_start)
    for i, metadata in enumerate(extractor_state_end["pdfs_metadata_extracted"]):
        print(f"pdfs_metadata_extracted[{i}]:")
        pprint(metadata)
        print("-" * 100)


# %%
class RandomRelevanceEvaluatorNode(IWorkflowNode):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __call__(self, state: RelevanceEvaluatorState) -> RelevanceEvaluatorState:
        logger.trace(f"{self.__class__.__name__}: __call__: start")
        pdfs = state["pdfs"]

        scores: List[RelevanceScoreContainer] = []
        for pdf in pdfs:
            score = round(random.random() * 10, 2)
            scores.append(
                RelevanceScoreContainer(scores=[RelevanceScore(score=score, criteria="random", comment="random")])
            )
        output = {
            "relevance_scores": scores,
        }
        logger.trace(f"{self.__class__.__name__}: __call__: end")
        return output


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
        logger.trace(f"{self.__class__.__name__}: __call__: start")

        user_query = state["user_query"]
        pdfs_metadata_extracted = state["pdfs_metadata_extracted"]

        response_schemas = get_response_schemas()
        prompt_template = get_prompt_template()
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_query", "title", "abstract", "keywords", "year"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | self._llm | output_parser

        logger.debug(f"count of pdfs_metadata_extracted: {len(pdfs_metadata_extracted)}")

        responses = chain.batch(
            inputs=[
                {
                    "user_query": user_query,
                    "abstract": getattr(metadata, "abstract", ""),
                    "keywords": getattr(metadata, "keywords", []),
                    "title": getattr(metadata, "title", ""),
                    "year": getattr(metadata, "year", ""),
                }
                for metadata in pdfs_metadata_extracted
            ],
            config=self._config,
        )
        relevance_scores = []
        for response in responses:
            container = RelevanceScoreContainer(
                scores=[
                    RelevanceScore(
                        score=Score(value=response["theme_score"]),
                        criteria=RelevanceCriteria.THEME,
                        comment="Score for the theme of the article. The score is a number between 0 and 100.",
                    ),
                    RelevanceScore(
                        score=Score(value=response["terminology_score"]),
                        criteria=RelevanceCriteria.TERMINOLOGY,
                        comment="Score for the terminology of the article. The score is a number between 0 and 100.",
                    ),
                    RelevanceScore(
                        score=Score(value=response["methodology_score"]),
                        criteria=RelevanceCriteria.METHODOLOGY,
                        comment="Score for the methodology of the article. The score is a number between 0 and 100.",
                    ),
                    RelevanceScore(
                        score=Score(value=response["practical_applicability_score"]),
                        criteria=RelevanceCriteria.PRACTICAL_APPLICABILITY,
                        comment="Score for the practical applicability of the article. The score is a number between 0 and 100.",
                    ),
                    RelevanceScore(
                        score=Score(value=response["novelty_and_relevance_score"]),
                        criteria=RelevanceCriteria.NOVELTY_AND_RELEVANCE,
                        comment="Score for the novelty and relevance of the article. The score is a number between 0 and 100.",
                    ),
                    RelevanceScore(
                        score=Score(value=response["fundamental_significance_score"]),
                        criteria=RelevanceCriteria.FUNDAMENTAL_SIGNIFICANCE,
                        comment="Score for the fundamental significance of the article. The score is a number between 0 and 100.",
                    ),
                ]
            )
            relevance_scores.append(container)

        output = {
            "relevance_scores": relevance_scores,
        }
        logger.trace(f"{self.__class__.__name__}: __call__: end")
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
            source_query="Менеджмент энергии в микросетке с использованием пакетного обучения с подкреплением",
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="3219819.3220096.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
            source_query="Менеджмент энергии в микросетке с использованием пакетного обучения с подкреплением",
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="Intl J Robust   Nonlinear - 2021 - Wan - Optimal control and learning for cyber‐physical systems.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
            source_query="Машинное обучение. Обучение с подкреплением",
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="main_thesis.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
            source_query="Я пишу диссертацию по теме: Обучение с подкреплением. Обучение в офлайн-режиме.",
        ),
        OpenAlexPDF(
            url="https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-021-00561-9",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="s13321-021-00561-9.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
    ]

    user_query = "Менеджмент энергии в микросетке с использованием пакетного обучения с подкреплением"

    evaluator_state_start = RelevanceEvaluatorState(
        user_query=HumanMessage(
            content=(f"Я пишу диссертацию по теме: {user_query}. Скачай все статьи по этой теме. \n/no-think"),
        ),
        pdfs=pdfs,
        pdfs_metadata_extracted=extractor_state_end["pdfs_metadata_extracted"],
        relevance_scores=[],
        works=None,
    )

    pdf_relevance_evaluator_node = PDFRelevanceEvaluatorNode(llm=llm)
    evaluator_state_end = pdf_relevance_evaluator_node(evaluator_state_start)
    pprint(evaluator_state_start["user_query"].content)
    for i, pdf in enumerate(evaluator_state_start["pdfs"]):
        pprint(pdf.filename)
        container_scores = evaluator_state_end["relevance_scores"][i]
        for relevance_score in container_scores:
            print(f"{relevance_score.criteria.value}: {relevance_score.score.value}")
        print(f"mean: {container_scores.mean:.2f}")
        print("-" * 100)


# %%
class RelevanceEvaluatorWorkflowBuilder(IWorkflowBuilder):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def build(self, **kwargs) -> StateGraph:
        logger.trace(f"{self.__class__.__name__}: build: start")
        graph_builder = StateGraph(RelevanceEvaluatorState)
        graph_builder.add_node("PDFMetadataExtractor", PDFMetadataExtractorNode(llm=self._llm))
        graph_builder.add_node("RelevanceEvaluator", PDFRelevanceEvaluatorNode(llm=self._llm))
        # graph_builder.add_node("RelevanceEvaluator", RandomRelevanceEvaluatorNode(llm=self._llm))
        graph_builder.add_edge(START, "PDFMetadataExtractor")
        graph_builder.add_edge("PDFMetadataExtractor", "RelevanceEvaluator")
        # graph_builder.add_edge(START, "RelevanceEvaluator")
        graph_builder.add_edge("RelevanceEvaluator", END)
        graph = graph_builder.compile(**kwargs)
        logger.trace(f"{self.__class__.__name__}: build: end")
        return graph


if __name__ == "__main__":
    # import os
    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # llm = ChatOllama(
    #     model="qwen3:8b",
    #     temperature=0.0,
    #     max_tokens=10000,
    # )
    from langchain.chat_models import ChatOpenAI

    from relepaper.domains.langgraph.workflows.utils.graph_displayer import (
        DisplayMethod,
        GraphDisplayer,
    )

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
            source_query="Менеджмент энергии в микросетке с использованием пакетного обучения с подкреплением",
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="3219819.3220096.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
            source_query="Менеджмент энергии в микросетке с использованием пакетного обучения с подкреплением",
        ),
        OpenAlexPDF(
            url="https://some-url.com",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="Intl J Robust   Nonlinear - 2021 - Wan - Optimal control and learning for cyber‐physical systems.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
            source_query="Машинное обучение. Обучение с подкреплением",
        ),
        OpenAlexPDF(
            url="https://dr.ntu.edu.sg/bitstream/10356/172831/2/main_thesis.pdf",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="main_thesis.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
            source_query="Я пишу диссертацию по теме: Обучение с подкреплением. Обучение в офлайн-режиме.",
        ),
        OpenAlexPDF(
            url="https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-021-00561-9",
            dirname=Path(
                "/home/rinkorn/space/prog/python/sber/project-relepaper/src/relepaper/domains/langgraph/workflows/.data/openalex_pdfs"
            ),
            filename="s13321-021-00561-9.pdf",
            strategy=PDFDownloadStrategy.SELENIUM,
        ),
    ]
    workflow = RelevanceEvaluatorWorkflowBuilder(llm=llm).build()
    displayer = GraphDisplayer(workflow).set_strategy(DisplayMethod.MERMAID)
    displayer.display()

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
            pdfs_metadata_extracted=[],
            relevance_scores=[],
            pdf_analyser_state=PDFAnalyserState(
                short_long_pdf_length_threshold=100000,
                max_chunk_length=100000,
                max_chunks_count=10,
                intersection_length=1000,
            ),
        )
        state_end = workflow.invoke(input=state_start)

        for pdf, extracted_metadata, pdf_container_score in zip(
            state_end["pdfs"],
            state_end["pdfs_metadata_extracted"],
            state_end["relevance_scores"],
        ):
            print("-" * 100)
            print(f"User query: {user_query}")
            print(f"Title: {getattr(extracted_metadata, 'title', '')}")
            print(f"PDF: {pdf.filename}")
            for relevance_score in pdf_container_score:
                print(relevance_score)
            print(f"Mean: {pdf_container_score.mean:.2f}")
            print("-" * 100)
