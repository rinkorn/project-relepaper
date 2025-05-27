# %%
import gradio as gr
from langchain.schema import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

# %%
model = ChatOllama(
    # model="qwen2.5-coder:14b",
    model="qwen3:8b",
    temperature=0.0,
    max_tokens=10000,
)


# %%
def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = model.invoke(history_langchain_format)
    return gpt_response.content


demo = gr.ChatInterface(
    predict,
    type="messages",
    title="Relepaper",
    description="Relepaper is a tool for evaluating the quality of your research papers.",
    theme="soft",
    examples=[
        ["Нужны статьи по теме 'Искусственный интеллект'"],
        ["Пишу диссертацию по теме 'Обучение с подкреплением. Dreamer'. Какие самые релевантные статьи по этой теме?"],
    ],
)


# %%
if __name__ == "__main__":
    demo.launch()
