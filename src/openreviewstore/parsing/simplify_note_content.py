# %%
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def simplify_note_content(note_dict: Dict) -> Dict:
    """
    Упрощает структуру содержимого заметки, извлекая значения из полей.

    В API v2 содержимое полей имеет структуру {field: {value: actual_value}},
    эта функция преобразует ее в {field: actual_value}.

    Args:
        note_dict: Словарь с данными заметки

    Returns:
        Словарь с упрощенной структурой содержимого
    """
    if "content" not in note_dict:
        return note_dict

    for key, value in note_dict["content"].items():
        if isinstance(value, dict) and "value" in value:
            note_dict["content"][key] = value["value"]

    return note_dict
