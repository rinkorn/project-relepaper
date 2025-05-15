# %%
# нужен агент, который будет:
#   - переформулировать идею/запрос пользователя на множество релевантных запросов с возможными модификациями(напримиер, смена ключевых слов, изменение формулировки, и т.д.)
#   - искать релевантные pdf в интернете через pyalex
#   - скачивать pdf и сохранять в папку
#   - сохранять информацию о pdf в базу данных
#   - обрабатывать pdf с помощью LLM
#   - сохранять обработанную информацию о pdf в базу данных
#   - выводить пользователю обработанную информацию о pdf


# %%
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI

# %%
llm = ChatOpenAI(
    base_url="http://localhost:7007/v1",
    api_key="not_needed",
    temperature=0.1,
    # max_tokens=512,
)

# %%
# создаем пайтон агента
agent_python_repl = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True,
)
# создаем sql агента
agent_executor_sql = create_sql_agent(
    llm=llm,
    db=SQLDatabase.from_uri("sqlite:///" + "./data_in/pizzeria.db"),
    verbose=True,
)
# оба агента станут инструментами для верхнеуровнего агента
tools_mix = [
    Tool(
        name="PythonAgent",
        func=agent_python_repl.run,
        description="""Useful to run python commands""",
    ),
    Tool(
        name="SQLAgent",
        func=agent_executor_sql.run,
        description="""Useful to query sql tables""",
    ),
]
# super агент
prompt = hub.pull("hwchase17/react")
agent_super = create_react_agent(
    tools=tools_mix,
    llm=llm,
    prompt=prompt,
)
agent_executor = AgentExecutor(
    agent=agent_super,
    tools=tools_mix,
    verbose=True,
    handle_parsing_errors=True,
)

query = "Возьми из таблицы orders данные о количестве заказов в день. И построй график для 10 дней."
response = agent_executor.invoke({"input": query})
print(response)


# %%
query = "Возьми из таблицы orders данные о количестве заказов в день. И построй график для 10 дней."
response = agent_executor.invoke({"input": query})
print(response)


# %%
query = """\
Возьми из таблицы orders данные о количестве заказов в день.
Построй график для первых 10 дней синим цветом.
Аппроксимируй эти значения полиномом 4го порядка и построй на этом же графике красным цветом.
На графике нужна регулярная сетка и подписи на русском языке.
"""

response = agent_executor.invoke({"input": query})
print(response)


# %%
query = """\
Возьми из таблицы pizzas все разновидности пицц. Построй график где сколько заказали каждую.

Построй ещё один график, где будет понятно сколько потратили денег на каждую разновидность пиццы.
"""
response = agent_executor.invoke({"input": query})
print(response)


# %%
query = """\
Выдели все разновидности пицц из таблицы pizzas.

Для клиентов из таблицы clients:
    - Якун Бабиков
    - Мирон Нехлюдов
    - Гордей Брагин
укажи сколько они заказали каждой разновидности пиццы.

Построй на одном плоте разные графики для указанных клиентов разными цветами, где по осям будут разновидности пицц и количество каждой.
Используй matplotlib, не используй seaborn.
"""
response = agent_executor.invoke({"input": query})
print(response)
# %%
