import os
import pickle
import logging

import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- Настройка логирования ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Константы и конфигурация ---
load_dotenv()
RAG_BOT_TOKEN = os.getenv("RAG_BOT_TOKEN")

MODEL_NAME = 'all-MiniLM-L6-v2'
STORE_PATH = 'vector_store'
INDEX_FILE = os.path.join(STORE_PATH, 'index.faiss')
MAP_FILE = os.path.join(STORE_PATH, 'message_map.pkl')

# Количество результатов для поиска
TOP_K = 3

# --- Глобальные переменные для хранения загруженных данных ---
# Загружаем модель и базу один раз при старте бота для скорости
model = None
faiss_index = None
message_map = None

# --- Функции-обработчики команд ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение при команде /start."""
    await update.message.reply_text(
        "Привет! Я бот, который поможет тебе найти информацию в архиве чата.\n"
        "Просто отправь мне свой вопрос, и я найду наиболее похожие сообщения."
    )

async def search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выполняет поиск по векторной базе."""
    user_query = update.message.text
    
    if not faiss_index or not message_map:
        await update.message.reply_text("База знаний еще пуста или не загружена. Попробуйте позже.")
        return

    logger.info(f"Получен поисковый запрос: '{user_query}'")

    # 1. Преобразуем запрос пользователя в вектор
    query_vector = model.encode([user_query])
    
    # 2. Ищем в FAISS K ближайших векторов
    # search возвращает расстояния (D) и индексы (I)
    distances, indices = faiss_index.search(np.array(query_vector).astype('float32'), TOP_K)

    # 3. Формируем ответ
    results = []
    found_indices = indices[0] # Индексы для первого (и единственного) запроса
    
    # Убираем результаты с -1, если их меньше чем TOP_K
    valid_indices = [i for i in found_indices if i != -1]
    
    if not valid_indices:
        await update.message.reply_text("К сожалению, ничего похожего не найдено.")
        return

    for i in valid_indices:
        # Получаем данные сообщения из нашей карты
        message_data = message_map.get(i)
        if message_data:
            results.append(message_data['text'])

    # Собираем все в одно сообщение
    response_text = "Вот что я нашел:\n\n"
    response_text += "\n\n---\n\n".join(f"🔹 {text}" for text in results)

    await update.message.reply_text(response_text)


def main() -> None:
    """Основная функция для запуска бота."""
    global model, faiss_index, message_map

    if not RAG_BOT_TOKEN:
        logger.error("Токен RAG-бота не найден! Проверьте .env файл.")
        return

    # --- Загрузка модели и векторной базы при старте ---
    try:
        logger.info("Загрузка ML-модели...")
        model = SentenceTransformer(MODEL_NAME)
        logger.info("ML-модель успешно загружена.")

        logger.info("Загрузка векторного хранилища...")
        faiss_index = faiss.read_index(INDEX_FILE)
        with open(MAP_FILE, 'rb') as f:
            message_map = pickle.load(f)
        logger.info(f"Хранилище успешно загружено. В индексе {faiss_index.ntotal} векторов.")

    except FileNotFoundError:
        logger.warning(
            "Файлы векторного хранилища не найдены. "
            "Бот будет работать, но поиск не даст результатов, пока сканер не создаст базу."
        )
    except Exception as e:
        logger.error(f"Произошла критическая ошибка при загрузке данных: {e}")
        return

    # --- Настройка и запуск самого бота ---
    application = Application.builder().token(RAG_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    # Отвечаем на все текстовые сообщения, которые не являются командами
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))

    logger.info("Запуск бота...")
    application.run_polling()


if __name__ == "__main__":
    main()