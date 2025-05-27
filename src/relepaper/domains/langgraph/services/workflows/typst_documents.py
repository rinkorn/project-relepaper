import json
import os
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_gigachat.chat_models import GigaChat
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from relepaper.config.dev_settings import get_dev_settings
from relepaper.domains.langgraph.services.tool_decorator import tool_decorator
from relepaper.domains.openalex.entities.openalex_work import OpenAlexWork

load_dotenv(find_dotenv())

SESSION_PATH = get_dev_settings().project_path / "20250526T074307_fcb81239bd594c498640b5d85386d67d/"


# %%
def load_openalex_works(dirname: Path) -> list[OpenAlexWork]:
    # Загружаем данные по работам
    works: list[OpenAlexWork] = []
    for work_id in dirname.glob("*.json"):
        with open(work_id, "r") as f:
            work = json.load(f)
        works.append(OpenAlexWork.from_external(work))
    return works


def generate_pdf_document(works: list[OpenAlexWork]) -> None:
    """
    Генерирует PDF, в котором заполнены краткие данные по работам

    Args:
        works (list[OpenAlexWork]): список работ для внесения в акт

    Returns:
        None
    """
    act_json = {"works": list(map(lambda w: asdict(w), works))}
    with open(os.path.join("typst", "document.json"), "w") as f:
        json.dump(act_json, f, ensure_ascii=False)
    command = ["typst", "compile", "--root", "./typst", "typst/document.typ"]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr)


# %%
tools = {
    "generate_pdf_document": tool_decorator(generate_pdf_document),
}


# %%
class LLMAgent:
    def __init__(self, model: LanguageModelLike, tools: dict[str, BaseTool]):
        self._model = model
        self._agent = create_react_agent(model, tools=tools, checkpointer=InMemorySaver())
        self._config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}

    def upload_file(self, file):
        file_uploaded_id = self._model.upload_file(file).id_  # type: ignore
        return file_uploaded_id

    def invoke(self, content: str, attachments: list[str] | None = None, temperature: float = 0.1) -> str:
        """Отправляет сообщение в чат"""
        message: dict = {"role": "user", "content": content, **({"attachments": attachments} if attachments else {})}
        return self._agent.invoke({"messages": [message], "temperature": temperature}, config=self._config)["messages"][
            -1
        ].content


def print_agent_response(llm_response: str) -> None:
    print(f"\033[35m{llm_response}\033[0m")


def get_user_prompt() -> str:
    return input("\nТы: ")


def main():
    model = GigaChat(
        model="GigaChat-2-Max",
        verify_ssl_certs=False,
    )

    agent = LLMAgent(model, tools=tools.values())
    system_prompt = (
        "You are a helpful assistant that can generate typst documents. "
        "You will be given a list of works and you will need to generate a typst document. "
        "The document will be a typst document. "
    )

    file_uploaded_id = agent.upload_file(
        open(SESSION_PATH / "openalex_works" / "W3106804518.json", "rb"),
    )
    agent_response = agent.invoke(
        content=system_prompt,
        attachments=[file_uploaded_id],
    )

    while True:
        print_agent_response(agent_response)
        user_prompt = get_user_prompt()
        agent_response = agent.invoke(user_prompt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nдосвидули!")
