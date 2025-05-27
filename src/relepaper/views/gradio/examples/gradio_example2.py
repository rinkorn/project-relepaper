# %%
import random
import time
from pprint import pprint

import gradio as gr
from dotenv import load_dotenv
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOllama(
    model="qwen2.5-coder:14b",
    temperature=0,
    max_tokens=10000,
    streaming=True,
)

# %%
# search = DuckDuckGoSearchRun()
# pprint(search.invoke("Кто такой ВВП"))

wrapper = DuckDuckGoSearchAPIWrapper(
    region="ru-ru",
    time="d",
    max_results=2,
)
search = DuckDuckGoSearchResults(
    api_wrapper=wrapper,
    output_format="list",
    backend="news",
)
pprint(search.invoke("Переговоры в Турции"))

# %%
tools = {}
tools["search_in_web"] = search
llm_with_tools = llm.bind_tools([t for t in tools.values()])


# %%
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_react_agent(
    model=llm,
    tools=[t for t in tools.values()],
    prompt="You are a helpful assistant",
    # checkpointer=checkpointer,
    # response_format=WeatherResponse,
)


# %%
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history: list):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        history.append({"role": "assistant", "content": ""})
        for character in bot_message:
            history[-1]["content"] += character
            time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

# %%
if __name__ == "__main__":
    demo.launch()
