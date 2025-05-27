import abc


class IPresenter(abc.ABC):
    """Базовый интерфейс для всех презентеров в паттерне MVP."""

    @abc.abstractmethod
    def run(self, user_input: str = None) -> str:
        """Запускает презентер с опциональным пользовательским вводом."""
        pass
