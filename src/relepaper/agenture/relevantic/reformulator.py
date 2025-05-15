# %% ----- Intro -----
# В LangChain представлены агенты 5 основных типов (`Agent Types`):
# - ReAct (Reasoning and Acting) - агенты такого типа создаются по умолчанию,
# работают с инструментами, принимающими один параметр на вход.
# - Structured chat - могут работать с инструментами, которые принимают много
# параметров на вход
# - Self Ask with Search - реализация техники промптинга `Self-Ask`, работает
# только с инструментами поиска
# - XMLAgent, JSONAgent - для работы с чат моделями в форматах `XML` и `JSON`
# (например, модели от `Anthropic`)
# - OpenAI Tools Agents - пришли на замену `OpenAIFunctions Agents` (OpenAI
# перестает их поддерживать). Могут вызывать сразу одну или несколько функций,
# тем самым сокращая время для получения ответа - работает с последними моделями
# от OpenAI.


# %% ----- ReAct -----
# Этот тип агента реализуется на основе техники `ReAct Prompting` представленной
# командой GoogleResearch в 2022 году в статье [ReAct: Synergizing Reasoning and
# Acting in Language Models](https://arxiv.org/abs/2210.03629).

# ReAct объединила в себе два подхода:
# - Reasoning (техника Chain of Thoughts)
# - Acting (Action Plan Generation - генерация плана действий)


# %%
from pprint import pprint

from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import StructuredTool, tool
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
# from langchain_openai import ChatOpenAI

# %%
# llm = ChatOpenAI(
#     base_url="http://localhost:7007/v1",
#     api_key="not_needed",
#     temperature=0.2,
# )
# llm = ChatOpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama",
#     model="qwen2.5-coder:14b",
# )


llm = ChatOllama(
    model="qwen2.5-coder:14b",
    temperature=0.01,
    # num_predict = 256,
    # other params ...
)

# llm = ChatOpenAI(
#     course_api_key=course_api_key,
# ).bind(
#     functions=[classifier_function],
#     function_call={"name": "TopicClassifier"},
# )


# %% ----- StructuredTool -----
# Создадим свой tool, вычисляющий площадь треугольника
class TriangleAreaArgs(TypedDict):
    a: Annotated[float, ..., "Длина стороны a"]
    b: Annotated[float, ..., "Длина стороны b"]
    c: Annotated[float, ..., "Длина стороны c"]


def compute_triangle_area(a: float, b: float, c: float) -> float:
    """Вычисляет площадь треугольника по длинам его сторон
    Args:
        a (float): длина стороны a
        b (float): длина стороны b
        c (float): длина стороны c
    Returns:
        Площадь треугольника
    """
    s = (a + b + c) / 2
    return (s * (s - a) * (s - b) * (s - c)) ** 0.5


# print(compute_triangle_area.name)
# print(compute_triangle_area.description)
# print(compute_triangle_area.args)
# print(compute_triangle_area.return_direct)

compute_triangle_area_tool = StructuredTool.from_function(compute_triangle_area)
compute_triangle_area_tool.invoke(TriangleAreaArgs(a=3.0, b=4.0, c=5.0))  # 6.0

# %%
# tools = [
#     Tool(
#         name="compute_triangle_area tool",
#         func=compute_triangle_area,
#         description="Вычисляет площадь треугольника по длинам его сторон.",
#         args_schema=TriangleAreaArgs,
#     ),
# ]
# tools = [
#     compute_triangle_area_tool,
# ]
tools = [
    tool(compute_triangle_area),
]
# pprint(tools[0])

llm = llm.bind_tools(tools)


# %%
prompt = hub.pull("hwchase17/structured-chat-agent")
# возьмем запрос и переведем на русский язык
prompt.messages[0].prompt.template = """\
Отвечайте человеку максимально услужливо и точно. 
У вас есть доступ к следующим инструментам:

{tools}

Используйте большой двоичный объект json, чтобы указать инструмент, указав ключ действия (tool name) и ключ action_input (tool input).

Правильные "action" значения: "Final Answer" или {tool_names}.

Укажите только ОДНО действие для каждого $JSON_BLOB, как показано:

```
{{
 "action": $TOOL_NAME,
 "action_input" : $INPUT
}}
```

Следуйте этому формату:

Вопрос: введите вопрос для ответа
Мысль: обдумайте предыдущие и последующие шаги
Действие:
```
$JSON_BLOB
```
Наблюдение: результат действия
... (повторите Мысль/Действие/Наблюдение N раз)
Мысль: Я знаю, что ответить
Действие:
```
{{
 "action": " Final Answer",
 "action_input": "Окончательный ответ человеку"
}}

Начинаем!
Напоминание: ВСЕГДА отвечайте действительным json-объектом одного действия.
При необходимости используйте инструменты.
Если уместно, ответьте прямо.
Формат: действие:```$JSON_BLOB```, затем наблюдение.
"""
pprint(prompt.messages[0].prompt.template)


agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,  # максимальное количество итераций агента (15 по умолчанию)
    # handle_parsing_errors=True,
)
agent_msg = agent_executor.invoke(
    {"input": "Посчитай площадь треугольника со сторонами 341232.0, 341232.0, 34526.0"},
)
pprint(agent_msg)
# %%
