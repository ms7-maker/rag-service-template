"""
Пути и имена очищенных документов: data/<stem>_clean.txt
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CLEAN_SUFFIX = "_clean"
CLEAN_GLOB = f"*{CLEAN_SUFFIX}.txt"


def stem_from_input_path(path: Path) -> str:
    """aroma.txt → aroma; aroma_clean.txt → aroma."""
    stem = path.stem
    if stem.endswith(CLEAN_SUFFIX):
        stem = stem[: -len(CLEAN_SUFFIX)]
    return stem or "document"


def stem_from_url(url: str) -> str:
    """https://example.com/ → example."""
    parsed = urlparse(url.strip())
    host = (parsed.netloc or parsed.path.split("/")[0]).lower()
    host = host.removeprefix("www.")
    if not host:
        return "page"
    label = host.split(".")[0] if "." in host else host
    slug = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return slug or "page"


def clean_output_path(stem: str, data_dir: Path | None = None) -> Path:
    """Имя файла: data/<stem>_clean.txt"""
    base = data_dir if data_dir is not None else DATA_DIR
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_") or "document"
    return base / f"{safe}{CLEAN_SUFFIX}.txt"


def list_clean_files(data_dir: Path | None = None) -> list[Path]:
    """Все очищенные txt в data/, отсортированные по имени."""
    base = data_dir if data_dir is not None else DATA_DIR
    if not base.is_dir():
        return []
    return sorted(base.glob(CLEAN_GLOB))


def load_documents_from_data(
    data_dir: Path | None = None,
) -> list[tuple[str, str]]:
    """
    Загрузить документы из data/*_clean.txt.

    Returns:
        Список (имя_файла, текст) для индексации.
    """
    docs: list[tuple[str, str]] = []
    for path in list_clean_files(data_dir):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append((path.name, text))
    return docs


def read_combined_clean_files(
    data_dir: Path | None = None,
    separator: str = "\n\n---\n\n",
) -> str:
    """Склеить все *_clean.txt (для обратной совместимости)."""
    parts = [text for _, text in load_documents_from_data(data_dir)]
    return separator.join(parts)
