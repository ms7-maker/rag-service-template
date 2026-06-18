"""
Заглушка fallback для публичного шаблона.

В приватном репозитории здесь: обработка низкой уверенности retrieval,
уточняющие вопросы, эскалация и запасные модели.
См. docs/PUBLIC_AND_PRIVATE.md
"""

from typing import Dict, Any, List, Optional


def should_fallback(
    context_docs: List[Dict[str, Any]],
    min_score: Optional[float] = None,
) -> bool:
    """Публичная версия: fallback отключён."""
    return False


def get_fallback_response(
    user_query: str,
    context_docs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Публичная версия: всегда None — ответ генерирует основной пайплайн."""
    return None
