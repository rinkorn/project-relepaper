from relepaper.domains.langgraph.services.interfaces import IService
from relepaper.presenters.interfaces import IPresenter
from relepaper.views.interfaces import IView


class GradioPresenter(IPresenter):
    def __init__(self, view: IView, chat_model_service: IService):
        self._view = view
        self._chat_model_service = chat_model_service

    def run(self) -> str:
        """Запускает Gradio интерфейс."""
        import gradio as gr
        from langchain_core.messages import AIMessage, HumanMessage

        def predict(message, history):
            history_langchain_format = []
            for msg in history:
                if msg["role"] == "user":
                    history_langchain_format.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history_langchain_format.append(AIMessage(content=msg["content"]))
            history_langchain_format.append(HumanMessage(content=message))
            response = self._chat_model_service.invoke(history_langchain_format)
            return response.content

        demo = gr.ChatInterface(
            predict,
            type="messages",
            title="Relepaper",
            description="Relepaper is a tool for evaluating the quality of your research papers.",
            theme="soft",
            examples=[
                ["Нужны статьи по теме 'Искусственный интеллект в горном деле'"],
                [
                    "Пишу диссертацию по теме 'Обучение с подкреплением. Dreamer'. Какие самые релевантные статьи по этой теме?"
                ],
            ],
        )
        demo.launch()

        self._view.display_chat()
