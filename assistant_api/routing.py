"""
Заглушка маршрутизации для публичного шаблона.

В приватном репозитории здесь: классификация намерений, выбор коллекции FAISS,
выбор модели и ветвление пайплайна. См. docs/PUBLIC_AND_PRIVATE.md
"""

from typing import Dict, Any


def route_query(user_query: str) -> Dict[str, Any]:
    """
    Определить, как обрабатывать запрос.

    Публичная версия: всегда стандартный RAG без ветвления.
    """
    return {
        "route": "rag",
        "collection": None,
        "model": None,
        "reason": "public template: no routing",
    }
