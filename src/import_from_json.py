import os
import json
import pickle
import logging

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Настройка логирования ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Константы (должны совпадать со scanner.py и bot.py) ---
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
STORE_PATH = 'vector_store'
INDEX_FILE = os.path.join(STORE_PATH, 'index.faiss')
MAP_FILE = os.path.join(STORE_PATH, 'message_map.pkl')

# Имя вашего файла с экспортом
INPUT_JSON_FILE = 'result.json' 

# --- Функции, скопированные из scanner.py для совместимости ---

def load_vector_store():
    """Загружает FAISS индекс и карту сообщений. Если их нет - создает новые."""
    os.makedirs(STORE_PATH, exist_ok=True)
    
    if os.path.exists(INDEX_FILE) and os.path.exists(MAP_FILE):
        logger.info("Загрузка существующего векторного хранилища для дополнения...")
        index = faiss.read_index(INDEX_FILE)
        with open(MAP_FILE, 'rb') as f:
            message_map = pickle.load(f)
        logger.info(f"Хранилище успешно загружено. В индексе уже {index.ntotal} векторов.")
    else:
        logger.info("Создание нового векторного хранилища...")
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        message_map = {}
        logger.info("Новое хранилище создано.")
        
    return index, message_map

def save_vector_store(index, message_map):
    """Сохраняет FAISS индекс и карту сообщений на диск."""
    logger.info("Сохранение векторного хранилища на диск...")
    faiss.write_index(index, INDEX_FILE)
    with open(MAP_FILE, 'wb') as f:
        pickle.dump(message_map, f)
    logger.info(f"Хранилище успешно сохранено. Общее количество векторов: {index.ntotal}.")

# --- Новая функция для парсинга сложного поля 'text' ---

def parse_message_text(text_field) -> str:
    """
    Парсит поле 'text' из JSON-экспорта Telegram.
    Оно может быть строкой или списком из строк и словарей.
    """
    if isinstance(text_field, str):
        return text_field
    
    if isinstance(text_field, list):
        parts = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                # Это может быть спойлер, ссылка, жирный текст и т.д.
                # Нас интересует только текстовое содержимое.
                parts.append(item['text'])
        return "".join(parts)
        
    return "" # Возвращаем пустую строку, если формат неизвестен

# --- Основная логика импорта ---

def main():
    """Главная функция для запуска импорта."""
    if not os.path.exists(INPUT_JSON_FILE):
        logger.error(f"Файл экспорта '{INPUT_JSON_FILE}' не найден. Пожалуйста, поместите его в корень проекта.")
        return

    # 1. Загружаем модель
    logger.info(f"Загрузка ML-модели '{MODEL_NAME}'... Это может занять время.")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("ML-модель успешно загружена.")

    # 2. Загружаем или создаем векторное хранилище
    index, message_map = load_vector_store()
    
    # 3. Читаем и парсим JSON
    logger.info(f"Чтение файла экспорта '{INPUT_JSON_FILE}'...")
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = data.get('messages', [])
    if not messages:
        logger.error("В файле не найдено поле 'messages' или оно пустое.")
        return

    texts_to_process = []
    message_metadata = [] # Будем хранить тут пары (id, text)
    
    logger.info("Парсинг сообщений из файла...")
    for message in messages:
        # Обрабатываем только обычные сообщения с текстовым содержимым
        if message.get('type') == 'message' and message.get('text'):
            parsed_text = parse_message_text(message['text'])
            
            # Пропускаем пустые сообщения после парсинга
            if parsed_text and parsed_text.strip():
                texts_to_process.append(parsed_text)
                message_metadata.append({'id': message['id'], 'text': parsed_text})

    if not texts_to_process:
        logger.info("Не найдено текстовых сообщений для индексации.")
        return

    logger.info(f"Найдено {len(texts_to_process)} сообщений для индексации. Начинаем создание векторов...")

    # 4. Создаем эмбеддинги (векторы) для всех текстов разом
    # show_progress_bar=True очень полезно для долгих операций
    embeddings = model.encode(texts_to_process, show_progress_bar=True, convert_to_tensor=False)
    
    # 5. Добавляем векторы и метаданные в наше хранилище
    logger.info("Добавление векторов в индекс FAISS...")
    index.add(np.array(embeddings).astype('float32'))
    
    # Обновляем карту {id_в_индексе: метаданные}
    start_index = len(message_map)
    for i, meta in enumerate(message_metadata):
        message_map[start_index + i] = meta

    # 6. Сохраняем результат
    save_vector_store(index, message_map)
    logger.info("Импорт из JSON успешно завершен!")

if __name__ == "__main__":
    main()