from relepaper.views.interfaces import IView


class ConsoleView(IView):
    """
    Консольное представление для работы с чат-моделью.
    """

    def display_welcome(self):
        print("\n\n>>> AI: Напишите область интереса или область исследования, чтобы начать работу:")

    def display_response(self, response: str):
        print("\n\n>>> AI: " + response)

    def display_goodbye(self):
        print("\n\n>>> AI: Всего хорошего!")

    def get_user_query(self) -> str:
        return input("\n\n>>> User: ")
