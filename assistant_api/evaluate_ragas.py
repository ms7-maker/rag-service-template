"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API для RAG и для метрик RAGAS.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings

# Метрики RAGAS 0.4.x
try:
    # Классы-метрики (современный способ)
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import ContextPrecision

    faithfulness = Faithfulness
    context_precision = ContextPrecision
except ImportError:
    try:
        # Коллекции (альтернативный публичный API)
        from ragas.metrics.collections import faithfulness, context_precision
    except ImportError:
        # Старый импорт как последний вариант
        from ragas.metrics import faithfulness, context_precision

# Answer Relevancy (legacy-метрика, совместимая с evaluate())
try:
    from ragas.metrics._answer_relevance import answer_relevancy
except ImportError:
    answer_relevancy = None

from rag_pipeline import RAGPipeline


# Тестовые вопросы для оценки RAG системы
EVALUATION_QUESTIONS = [
    "Какие ароматы свечей есть?",
    "Сколько стоит доставка по Москве?",
    "Могу ли я оплатить курьеру наличными?",
    "Где вы находитесь?"
]


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов.
    
    Args:
        pipeline: RAG pipeline для получения ответов
        questions: список вопросов для оценки
    
    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("[*] Получение ответов от RAG системы...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")
        
        # Получаем ответ от RAG системы (без использования кеша)
        result = pipeline.query(question, use_cache=False)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - список текстов из найденных документов
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        
        # Ground truth - эталонный ответ (для демонстрации используем часть ответа)
        # В реальном проекте здесь должны быть вручную подготовленные эталонные ответы
        ground_truths_list.append(result["answer"][:100])
        
        print(f"     [+] Ответ получен от OpenAI API")
    
    print()
    
    # Создаём датасет для RAGAS
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    
    Процесс:
    1. Инициализация RAG pipeline
    2. Генерация ответов на тестовые вопросы
    3. Подготовка датасета для RAGAS
    4. Запуск оценки метрик
    5. Вывод результатов
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()

    # Внутри функции используем отдельную переменную, чтобы не было конфликтов областей видимости
    # legacy answer_relevancy импортируется на уровне модуля (может быть None)
    answer_relevancy_metric = answer_relevancy
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nУстановите переменную окружения:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        print("\nИли создайте файл .env в корне проекта с содержимым:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Инициализация RAG pipeline
    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_file="data/docs.txt",
            model="gpt-4o-mini"
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)
    
    # Подготовка датасета
    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    print("=" * 70)
    
    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision", end="")
    if answer_relevancy_metric is not None:
        print(", Answer Relevancy")
    else:
        print("\n   (Answer Relevancy недоступна в установленной версии RAGAS)")
    print("   (это займёт 1-2 минуты, так как RAGAS использует OpenAI для оценки)\n")

    # Готовим список метрик
    metrics_to_use = [faithfulness(), context_precision()]

    # Явно задаём LLM и эмбеддинги для RAGAS
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        ragas_llm = llm_factory("gpt-4o-mini", client=client)
        # Для legacy Answer Relevancy нужен LangChain-объект embeddings с embed_query/embed_documents
        ragas_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    except Exception as e:
        print(f"[ОШИБКА] Не удалось инициализировать LLM/эмбеддинги для RAGAS: {e}")
        sys.exit(1)

    # Запускаем оценку RAGAS (основные метрики одним прогоном)
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)

    # Какие метрики реально вернулись в результате
    returned_metrics = []
    try:
        if getattr(result, "scores", None) and isinstance(result.scores, list) and result.scores:
            returned_metrics = sorted(result.scores[0].keys())
    except Exception:
        returned_metrics = []
    if returned_metrics:
        print(f"[INFO] Метрики в результате: {', '.join(returned_metrics)}")

    # Если Answer Relevancy не посчиталась (нет колонки и/или NaN), попробуем получить причину
    # на маленьком подмножестве с raise_exceptions=True.
    if answer_relevancy_metric is not None:
        try:
            ar_series_debug = result["answer_relevancy"]
        except KeyError:
            ar_series_debug = None

        import math

        should_debug = False
        if ar_series_debug is None:
            should_debug = True
            nan_count = None
            total_count = None
        else:
            total_count = len(ar_series_debug)
            nan_count = sum(
                1 for v in ar_series_debug if isinstance(v, float) and math.isnan(v)
            )
            should_debug = nan_count > 0

        if should_debug:
            print()
            if nan_count is None:
                print(
                    "[WARN] Answer Relevancy не вернула колонку результата. "
                    "Пытаемся получить причину..."
                )
            else:
                print(
                    "[WARN] Answer Relevancy не вычислилась для части строк "
                    f"(NaN: {nan_count}/{total_count}). Пытаемся получить причину..."
                )
            try:
                debug_ds = dataset.select(range(min(1, len(dataset))))
                debug_result = evaluate(
                    dataset=debug_ds,
                    metrics=[answer_relevancy_metric],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    raise_exceptions=True,
                    show_progress=False,
                )
                try:
                    debug_vals = debug_result["answer_relevancy"]
                except KeyError:
                    debug_vals = None
                if debug_vals is None:
                    print("[DEBUG] Debug-прогон: колонка answer_relevancy не появилась")
                else:
                    print(f"[DEBUG] Debug-прогон: answer_relevancy={debug_vals}")
            except Exception as debug_exc:
                print(f"[DEBUG] Причина Answer Relevancy: {debug_exc}")

    # Если в общем прогоне answer_relevancy отсутствует (или вся NaN),
    # посчитаем её отдельным запуском evaluate() только для этой метрики.
    answer_relevancy_series = None
    if answer_relevancy_metric is not None:
        try:
            answer_relevancy_series = result["answer_relevancy"]
        except KeyError:
            answer_relevancy_series = None

        import math

        need_separate_run = False
        if answer_relevancy_series is None:
            need_separate_run = True
        else:
            nan_count = sum(
                1
                for v in answer_relevancy_series
                if isinstance(v, float) and math.isnan(v)
            )
            need_separate_run = nan_count == len(answer_relevancy_series)

        if need_separate_run:
            print("[INFO] Считаем Answer Relevancy отдельным прогоном...")
            try:
                ar_only = evaluate(
                    dataset=dataset,
                    metrics=[answer_relevancy_metric],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    raise_exceptions=False,
                )
                try:
                    answer_relevancy_series = ar_only["answer_relevancy"]
                except KeyError:
                    answer_relevancy_series = None
            except Exception as e:
                print(f"[WARN] Не удалось посчитать Answer Relevancy отдельно: {e}")
                answer_relevancy_series = None
    
    # Обработка и вывод результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    
    # Вычисляем средние значения метрик (игнорируя NaN)
    import math
    
    faithfulness_values = [
        v
        for v in result["faithfulness"]
        if not (isinstance(v, float) and math.isnan(v))
    ]
    context_precision_values = [
        v
        for v in result["context_precision"]
        if not (isinstance(v, float) and math.isnan(v))
    ]
    answer_relevancy_values = []
    if answer_relevancy_metric is not None:
        ar_series = answer_relevancy_series
        if ar_series is not None:
            answer_relevancy_values = [
                v
                for v in ar_series
                if not (isinstance(v, float) and math.isnan(v))
            ]
    
    avg_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values) 
        if faithfulness_values else 0
    )
    avg_context_precision = (
        sum(context_precision_values) / len(context_precision_values)
        if context_precision_values
        else 0
    )
    avg_answer_relevancy = (
        sum(answer_relevancy_values) / len(answer_relevancy_values)
        if answer_relevancy_values
        else 0
    )
    
    # Выводим общие метрики
    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    if answer_relevancy_metric is not None:
        print(
            f"   Answer Relevancy (релевантность ответа вопросу): "
            f"{avg_answer_relevancy:.4f}"
        )
    
    # Вычисляем и выводим средний балл
    # В среднем учитываем только те метрики, которые реально посчитались
    components = [avg_faithfulness, avg_context_precision]
    if answer_relevancy_metric is not None:
        components.append(avg_answer_relevancy)
    avg_score = sum(components) / len(components) if components else 0
    print(f"\n{'-'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")
    
    # Оценка качества системы
    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
        print("   Система показывает высокую точность и релевантность ответов.")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить качество документов или промптов.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Необходимо пересмотреть стратегию chunking или качество данных.")
    
    # Выводим детали по каждому вопросу
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)
    
    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\n{i+1}. {question}")
        
        # Faithfulness
        faith_val = result['faithfulness'][i]
        if not (isinstance(faith_val, float) and math.isnan(faith_val)):
            print(f"   Faithfulness:       {faith_val:.4f}")
        else:
            print(f"   Faithfulness:       не удалось вычислить")
        
        # Context Precision
        cp_val = result["context_precision"][i]
        if not (isinstance(cp_val, float) and math.isnan(cp_val)):
            print(f"   Context Precision:  {cp_val:.4f}")
        else:
            print(f"   Context Precision:  не удалось вычислить")

        # Answer Relevancy (если есть)
        if answer_relevancy_metric is not None:
            try:
                ar_val = (
                    answer_relevancy_series[i]
                    if answer_relevancy_series is not None
                    else None
                )
            except Exception:
                ar_val = None
            if ar_val is not None and not (
                isinstance(ar_val, float) and math.isnan(ar_val)
            ):
                print(f"   Answer Relevancy:   {ar_val:.4f}")
            elif ar_val is None:
                print("   Answer Relevancy:   метрика не считалась")
            else:
                print("   Answer Relevancy:   не удалось вычислить")
    
    # Пояснения к метрикам
    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness (Точность ответа):
  Измеряет, насколько ответ соответствует предоставленному контексту.
  Значения: 0.0 - 1.0 (1.0 = полное соответствие контексту)

Context Precision (Точность контекста):
  Измеряет качество извлечённого контекста для ответа на вопрос.
  Значения: 0.0 - 1.0 (1.0 = идеальный контекст)

Answer Relevancy (Релевантность ответа вопросу):
  Измеряет, насколько ответ по смыслу соответствует исходному вопросу.
  В вашем окружении метрика использует те же OpenAI LLM и эмбеддинги,
  что и Faithfulness/Context Precision.
    """)
    
    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()

