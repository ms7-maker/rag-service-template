"""
Модуль работы с векторным хранилищем FAISS.
Chunking, батч-эмбеддинги OpenAI, метаданные источника, поиск.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from data_paths import DATA_DIR, load_documents_from_data

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

EMBEDDING_DIM = 1536  # text-embedding-3-small
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100
DEFAULT_TEST_QUERY = "Какая информация содержится в базе знаний?"


class VectorStore:
    """Векторное хранилище на основе FAISS."""

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./faiss_db",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.store_path = os.path.join(persist_directory, collection_name)
        self.index_path = os.path.join(self.store_path, "index.faiss")
        self.metadata_path = os.path.join(self.store_path, "metadata.pkl")

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.index: faiss.Index | None = None
        self.documents: List[str] = []
        self.ids: List[str] = []
        self.sources: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

        os.makedirs(self.store_path, exist_ok=True)
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                meta = pickle.load(f)
            self.documents = meta["documents"]
            self.ids = meta.get("ids", [f"chunk_{i}" for i in range(len(self.documents))])
            n = len(self.documents)
            self.sources = meta.get("sources", ["unknown"] * n)
            self.metadatas = meta.get("metadatas", [{}] * n)
            print(
                f"Коллекция '{self.collection_name}' загружена. "
                f"Чанков: {self.count()}"
            )
        else:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            print(f"Создана новая коллекция '{self.collection_name}'")

    def count(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def reset(self) -> None:
        """Удалить индекс и метаданные (пересборка после смены документов)."""
        for path in (self.index_path, self.metadata_path):
            if os.path.exists(path):
                os.remove(path)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.documents = []
        self.ids = []
        self.sources = []
        self.metadatas = []
        print(f"Коллекция '{self.collection_name}' сброшена")

    def clear_collection(self) -> None:
        """Синоним reset()."""
        self.reset()

    def _save_index(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "ids": self.ids,
                    "sources": self.sources,
                    "metadatas": self.metadatas,
                },
                f,
            )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk = (
                    f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
                )
            elif current_chunk:
                chunks.append(current_chunk)
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = (
                    f"{overlap_text}\n\n{paragraph}" if overlap_text else paragraph
                )
            else:
                if len(paragraph) > chunk_size:
                    sentence_chunks = self._split_long_paragraph(
                        paragraph, chunk_size, overlap
                    )
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        return [chunk for chunk in chunks if len(chunk) >= 50]

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        if len(text) <= overlap_size:
            return text
        overlap_candidate = text[-overlap_size:]
        best_start = 0
        for delimiter in (". ", "! ", "? ", "\n"):
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        return overlap_candidate.strip()

    def _split_long_paragraph(
        self, paragraph: str, chunk_size: int, overlap: int
    ) -> List[str]:
        sentences = re.split(r"([.!?]+\s+)", paragraph)
        full_sentences: list[str] = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])

        chunks: list[str] = []
        current_chunk = ""
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = (
                        f"{overlap_text} {sentence}".strip() if overlap_text else sentence
                    )
                else:
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.openai_client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL,
        )
        return [item.embedding for item in response.data]

    def _embed_chunks_batched(self, chunks: List[str]) -> np.ndarray:
        all_embeddings: list[list[float]] = []
        total = len(chunks)
        for i in range(0, total, EMBEDDING_BATCH_SIZE):
            batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
            end = min(i + EMBEDDING_BATCH_SIZE, total)
            print(f"  Эмбеддинги: чанки {i + 1}-{end} из {total}")
            all_embeddings.extend(self._create_embeddings_batch(batch))
        vectors = np.array(all_embeddings, dtype=np.float32)
        return self._normalize(vectors)

    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Добавить документы: список (имя_источника, текст).

        Каждый документ режется на чанки; у каждого чанка есть source в метаданных.
        """
        if self.count() > 0:
            print("Документы уже загружены в коллекцию")
            return
        if not documents:
            raise ValueError("Список документов пуст")

        all_chunks: list[str] = []
        chunk_sources: list[str] = []
        chunk_metas: list[dict[str, Any]] = []

        print(f"Добавление {len(documents)} документ(ов)...")
        for doc_name, doc_text in documents:
            chunks = self._chunk_text(doc_text)
            print(f"  {doc_name}: {len(chunks)} чанков")
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_sources.append(doc_name)
                chunk_metas.append(
                    {"source": doc_name, "chunk_length": len(chunk)}
                )

        if not all_chunks:
            raise ValueError("После разбиения не осталось чанков для индексации")

        print(f"Создание эмбеддингов для {len(all_chunks)} чанков ({EMBEDDING_MODEL})...")
        vectors = self._embed_chunks_batched(all_chunks)
        self.index.add(vectors)

        base_id = 0
        for i, chunk in enumerate(all_chunks):
            self.documents.append(chunk)
            self.sources.append(chunk_sources[i])
            self.metadatas.append(chunk_metas[i])
            self.ids.append(f"chunk_{base_id + i}")

        self._save_index()
        print(
            f"Загружено {len(all_chunks)} чанков в коллекцию '{self.collection_name}'"
        )

    def load_from_data_dir(self, data_dir: Path | None = None) -> None:
        """Загрузить все data/*_clean.txt с сохранением источника по имени файла."""
        docs = load_documents_from_data(data_dir)
        if not docs:
            raise FileNotFoundError(
                f"Нет файлов {DATA_DIR}/*_clean.txt. "
                "Запустите cleaner.py или render_and_clean.py."
            )
        self.add_documents(docs)

    def load_text(self, text: str, source_label: str = "documents") -> None:
        """Один текстовый блок (обратная совместимость)."""
        self.add_documents([(source_label, text)])

    def load_documents(self, file_path: str) -> None:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Файл {file_path} не найден")
        text = path.read_text(encoding="utf-8")
        self.add_documents([(path.name, text)])

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.count() == 0:
            return []

        query_vec = self._normalize(
            np.array(self._create_embeddings_batch([query]), dtype=np.float32)
        )
        k = min(top_k, self.count())
        scores, indices = self.index.search(query_vec, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                {
                    "id": self.ids[idx],
                    "text": self.documents[idx],
                    "source": self.sources[idx],
                    "metadata": self.metadatas[idx],
                    "distance": float(1.0 - score),
                }
            )
        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        unique_sources = sorted(set(self.sources)) if self.sources else []
        return {
            "name": self.collection_name,
            "count": self.count(),
            "persist_directory": self.persist_directory,
            "engine": "faiss",
            "sources": unique_sources,
        }


if __name__ == "__main__":
    import sys

    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)

    test_query = os.getenv("VECTOR_STORE_TEST_QUERY", DEFAULT_TEST_QUERY).strip()
    if not test_query:
        test_query = DEFAULT_TEST_QUERY

    store = VectorStore(collection_name="test_collection")

    try:
        if store.count() == 0:
            store.load_from_data_dir()
    except FileNotFoundError as exc:
        print(f"Предупреждение: {exc}")
        print("Положите *_clean.txt в assistant_api/data/ и запустите снова.")
        sys.exit(1)

    print(f"\nТестовый запрос: {test_query}")
    results = store.search(test_query, top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        preview = doc["text"][:200] + ("..." if len(doc["text"]) > 200 else "")
        print(f"\n{i}. [{doc['source']}] {preview}")
        print(f"   distance: {doc['distance']:.4f}")

    print(f"\nСтатистика: {store.get_collection_stats()}")
