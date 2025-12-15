import os
import time
import pickle
import asyncio
import logging

import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from telegram import Bot
from telegram.error import TelegramError

# --- Настройка логирования ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Константы ---
# Загружаем переменные окружения из .env файла
load_dotenv()

SCANNER_BOT_TOKEN = os.getenv("SCANNER_BOT_TOKEN")
TARGET_CHAT_ID = int(os.getenv("TARGET_CHAT_ID"))

# Модель для создания эмбеддингов. all-MiniLM-L6-v2 - быстрая и качественная.
MODEL_NAME = 'all-MiniLM-L6-v2'
# Размерность векторов для этой модели
EMBEDDING_DIM = 384

# Пути для хранения данных
STORE_PATH = 'vector_store'
INDEX_FILE = os.path.join(STORE_PATH, 'index.faiss')
MAP_FILE = os.path.join(STORE_PATH, 'message_map.pkl')
STATE_FILE = os.path.join(STORE_PATH, 'last_message_id.txt')

# Периодичность сканирования в секундах
SCAN_INTERVAL_SECONDS = 10

# --- Функции управления состоянием и хранилищем ---

def get_last_message_id():
    """Читает ID последнего обработанного сообщения из файла."""
    try:
        with open(STATE_FILE, 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0  # Если файла нет, начинаем с самого начала

def save_last_message_id(message_id):
    """Сохраняет ID последнего обработанного сообщения в файл."""
    os.makedirs(STORE_PATH, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        f.write(str(message_id))

def load_vector_store():
    """Загружает FAISS индекс и карту сообщений. Если их нет - создает новые."""
    os.makedirs(STORE_PATH, exist_ok=True)
    
    if os.path.exists(INDEX_FILE) and os.path.exists(MAP_FILE):
        logger.info("Загрузка существующего векторного хранилища...")
        index = faiss.read_index(INDEX_FILE)
        with open(MAP_FILE, 'rb') as f:
            message_map = pickle.load(f)
        logger.info(f"Хранилище успешно загружено. В индексе {index.ntotal} векторов.")
    else:
        logger.info("Создание нового векторного хранилища...")
        # Создаем плоский L2 индекс. Он хорошо подходит для большинства задач.
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        # Карта будет хранить: {id_в_индексе: (id_сообщения_в_тг, текст)}
        message_map = {}
        logger.info("Новое хранилище создано.")
        
    return index, message_map

def save_vector_store(index, message_map):
    """Сохраняет FAISS индекс и карту сообщений на диск."""
    logger.info("Сохранение векторного хранилища на диск...")
    faiss.write_index(index, INDEX_FILE)
    with open(MAP_FILE, 'wb') as f:
        pickle.dump(message_map, f)
    logger.info("Хранилище успешно сохранено.")


# --- Основной цикл сканера ---

async def scan_chat():
    """Основная асинхронная функция для сканирования чата."""
    if not SCANNER_BOT_TOKEN:
        logger.error("Токен бота-сканера не найден! Проверьте .env файл.")
        return

    bot = Bot(token=SCANNER_BOT_TOKEN)
    logger.info("Загрузка ML-модели... Это может занять некоторое время.")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("ML-модель успешно загружена.")

    index, message_map = load_vector_store()
    
    while True:
        last_processed_id = get_last_message_id()
        logger.info(f"Начинаем сканирование с сообщения после ID: {last_processed_id}")

        try:
            # Получаем апдейты. offset - это ID, *начиная* с которого мы хотим получать сообщения
            # Telegram вернет сообщения с ID >= offset. Нам нужно last_processed_id + 1
            updates = await bot.get_updates(offset=last_processed_id + 1, timeout=20, allowed_updates=["message"])
        except TelegramError as e:
            logger.error(f"Ошибка при получении апдейтов от Telegram: {e}")
            time.sleep(60) # В случае ошибки ждем минуту
            continue

        if not updates:
            logger.info("Новых сообщений не найдено.")
        else:
            texts_to_process = []
            message_ids_to_process = []
            new_last_id = last_processed_id

            for update in updates:
                # Нас интересуют только сообщения в целевом чате
                if update.message and update.message.chat_id == TARGET_CHAT_ID:
                    if update.message.text: # Обрабатываем только текстовые сообщения
                        texts_to_process.append(update.message.text)
                        message_ids_to_process.append(update.message.message_id)

                # update.update_id - это уникальный ID самого события (апдейта), а не сообщения
                # Нам нужно отслеживать message_id, но для сдвига offset'а используется update_id
                # Чтобы не усложнять, будем использовать message_id, это тоже будет работать
                if update.message:
                    new_last_id = max(new_last_id, update.message.message_id)

            if texts_to_process:
                logger.info(f"Найдено {len(texts_to_process)} новых сообщений для индексации.")
                
                # 1. Генерируем эмбеддинги для всех новых сообщений пачкой
                embeddings = model.encode(texts_to_process, convert_to_tensor=False)
                
                # 2. Добавляем векторы в FAISS индекс
                index.add(np.array(embeddings).astype('float32'))
                
                # 3. Обновляем нашу карту
                start_index = len(message_map)
                for i, (text, msg_id) in enumerate(zip(texts_to_process, message_ids_to_process)):
                    message_map[start_index + i] = {"text": text, "id": msg_id}

                # 4. Сохраняем все на диск
                save_vector_store(index, message_map)
                save_last_message_id(new_last_id)
                logger.info(f"Индексация завершена. Последний ID сообщения: {new_last_id}")
            else:
                 # Если были апдейты, но не текстовые (вступление в чат и т.д.), все равно сдвигаем ID
                if new_last_id > last_processed_id:
                     save_last_message_id(new_last_id)
                logger.info("Новых *текстовых* сообщений в целевом чате не найдено.")

        logger.info(f"Следующее сканирование через {SCAN_INTERVAL_SECONDS} секунд.")
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        asyncio.run(scan_chat())
    except KeyboardInterrupt:
        logger.info("Сканер остановлен вручную.")