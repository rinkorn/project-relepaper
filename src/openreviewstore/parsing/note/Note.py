from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union


class AbstractNote(ABC):
    """Интерфейс для всех типов заметок."""

    def __init__(self, data: Dict[str, Any]):
        self._properties = data

    @abstractmethod
    def get_property(self, property_name: str, default: Any = None) -> Any:
        """Получить значение свойства заметки.

        Args:
            property_name: Имя свойства
            default: Значение по умолчанию, если свойство не найдено

        Returns:
            Значение свойства или default
        """
        pass

    def __str__(self) -> str:
        """Строковое представление заметки."""
        pass

    def __repr__(self) -> str:
        """Строковое представление заметки."""
        pass


class NoteAPIv1(AbstractNote):
    """Реализация заметки для API версии 1."""

    def get_property(self, property_name: str, default: Any = None) -> Any:
        """Получает значение свойства из данных заметки.

        В API v1 свойства могут находиться как на первом уровне, так и
        глубоко во вложенных структурах.

        Args:
            property_name: Имя свойства для поиска
            default: Значение по умолчанию, если свойство не найдено

        Returns:
            Значение свойства или default
        """
        found, value = self._find_property_recursive(property_name)
        return value if found else default

    def _find_property_recursive(self, property_name: str, data: Any = None) -> Tuple[bool, Any]:
        """"""
        if data is None:
            data = self._properties

        for key, value in data.items():
            if key == property_name:
                return True, value
            elif isinstance(value, dict):
                found, result = self._find_property_recursive(property_name, value)
                if found:
                    return True, result
        return False, None

    def __repr__(self) -> str:
        """Формирует строковое представление объекта с читаемым форматированием вложенных структур."""
        class_name = self.__class__.__name__

        def recursive_repr(obj, indent=0):
            indent_str = " " * indent

            if isinstance(obj, dict):
                if not obj:  # Пустой словарь
                    return "{}"

                parts = []
                for key, value in obj.items():
                    # Форматируем ключ
                    key_str = f"{indent_str}{key}"

                    # Обрабатываем значение в зависимости от его типа
                    if isinstance(value, Union[dict, list]) and value:
                        # Для сложных типов добавляем перевод строки и увеличиваем отступ
                        value_str = f":\n{recursive_repr(value, indent + 2)}"
                    else:
                        # Простые типы и пустые коллекции на той же строке
                        if isinstance(value, dict) and not value:
                            value_str = ": {}"
                        elif isinstance(value, list) and not value:
                            value_str = ": []"
                        elif isinstance(value, str) and len(value) > 50:
                            value_str = f": {value[:47]}..."
                        else:
                            value_str = f": {value}"

                    parts.append(f"{key_str}{value_str}")
                return "\n".join(parts)

            elif isinstance(obj, list):
                if not obj:  # Пустой список
                    return "[]"

                parts = []
                for i, item in enumerate(obj):
                    if isinstance(item, Union[dict, list]) and item:
                        # Для сложных типов - отдельная строка с отступом
                        item_str = f"{indent_str}- {i}:\n{recursive_repr(item, indent + 2)}"
                    else:
                        # Простые типы и пустые коллекции на той же строке
                        if isinstance(item, dict) and not item:
                            item_str = f"{indent_str}- {i}: {{}}"
                        elif isinstance(item, list) and not item:
                            item_str = f"{indent_str}- {i}: []"
                        else:
                            item_repr = str(item)
                            if len(item_repr) > 50:
                                item_str = f"{indent_str}- {i}: {item_repr[:47]}..."
                            else:
                                item_str = f"{indent_str}- {i}: {item_repr}"
                    parts.append(item_str)
                return "\n".join(parts)

            else:
                # Для простых типов просто возвращаем строковое представление
                return f"{indent_str}{obj}"

        properties_str = recursive_repr(self._properties)
        return f"{class_name}(\n{properties_str}\n)"

    def __str__(self) -> str:
        return self.__repr__()


class NoteAPIv2(AbstractNote):
    """Реализация заметки для API версии 2.

    Особенности API v2:
    - Свойства обычно хранятся на верхнем уровне
    - Значения свойств часто имеют формат {"value": actual_value}
    - Некоторые свойства могут быть вложенными в другие структуры
    """

    def get_property(self, property_name: str, default: Any = None) -> Any:
        """Получает значение свойства из данных заметки.

        В API v2 свойства обычно находятся на первом уровне в формате:
        {"property_name": {"value": actual_value}}

        Args:
            property_name: Имя свойства для поиска
            default: Значение по умолчанию, если свойство не найдено

        Returns:
            Значение свойства или default
        """
        # Сначала проверяем на основном уровне (быстрый путь)
        try:
            if property_name in self._properties:
                prop_value = self._properties[property_name]
                # Специфичный формат API v2 - словарь с ключом "value"
                if isinstance(prop_value, dict) and "value" in prop_value:
                    return prop_value["value"]
                return prop_value
        except (KeyError, TypeError):
            pass

        # Если не нашли, делаем полный рекурсивный поиск
        found, value = self._find_property_recursive(property_name)
        return value if found else default


class NoteFactory:
    """Фабрика для создания объектов заметок."""

    @staticmethod
    def create(data: Dict[str, Any]) -> AbstractNote:
        """Создать заметку соответствующего типа.

        Args:
            data: Словарь с данными заметки, включая api_version

        Returns:
            Объект заметки соответствующего типа

        Raises:
            ValueError: Если api_version отсутствует или неизвестна
        """
        if "api_version" not in data:
            raise ValueError("api_version not found in data")

        # Преобразуем api_version к строке для безопасного сравнения
        api_version = str(data["api_version"])

        if api_version == "1" or api_version == "v1":
            return NoteAPIv1(data)
        elif api_version == "2" or api_version == "v2":
            return NoteAPIv2(data)
        else:
            raise ValueError(f"Unknown api_version: {data['api_version']}")


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Файлы для тестирования, которые содержат details.replies
    store_path = Path("/home/rinkorn/space/prog/python/sber/project-openreviewstore/data/test_store")
    note_api = store_path / "ISMIR.net/2018/WoRMS/note/H1ll_CVUQ7.json"
    # note_api = store_path / "ICLR.cc/2025/Workshop/SynthData/note/6DV6DCk8GS.json"

    with open(note_api, "r") as f:
        data = json.load(f)

    # Используем фабрику
    note = NoteFactory.create(data)
    print(note)
    print(note.get_property("id"))
    print(note.get_property("title"))
    print(note.get_property("content"))
    print(note.get_property("details"))
