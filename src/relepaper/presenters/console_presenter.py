from relepaper.domains.langgraph.services.chat_model_service import ChatModelService
from relepaper.presenters.interfaces import IPresenter
from relepaper.views.interfaces import IView


class ConsolePresenter(IPresenter):
    def __init__(
        self,
        view: IView,
        chat_model_service: ChatModelService,
    ):
        self._view = view
        self._chat_model_service = chat_model_service

    def run(self):
        self._view.display_welcome()
        while True:
            query = self._view.get_user_query()
            if query == "exit" or query == "quit" or query == "q":
                break
            self._run_workflow(query)
        self._view.display_goodbye()

    def _run_workflow(self, query: str):
        response = self._chat_model_service.invoke(query)
        self._view.display_response(response.content)
