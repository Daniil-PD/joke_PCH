import sqlite3
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.schema import BaseOutputParser
from langchain.output_parsers import StructuredOutputParser
from langchain.graphs import LangGraph, Node

DB_PATH = "jokes.db"

class JokeInfoParser(BaseOutputParser):
    def parse(self, text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)
    conn.commit()
    conn.close()

def get_all_categories() -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM categories")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def get_or_create_category(category_name: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM categories WHERE name = ?", (category_name,))
    row = cursor.fetchone()
    if row:
        category_id = row[0]
    else:
        cursor.execute("INSERT INTO categories (name) VALUES (?)", (category_name,))
        conn.commit()
        category_id = cursor.lastrowid
    conn.close()
    return category_name

def create_is_joke_node():
    prompt_template = """
Ты — ассистент, который анализирует текст и определяет, является ли он шуткой или анекдотом.
Ответь JSON с полем:
{
    "is_joke": true или false
}

Текст для анализа:
{text}
"""
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    llm = OpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    output_parser = StructuredOutputParser(pydantic_object=JokeInfoParser)
    return Node(chain=chain, output_parser=output_parser)

def create_joke_info_node(existing_categories):
    categories_str = ", ".join(existing_categories) if existing_categories else "Нет категорий"
    prompt_template = f"""
Ты — ассистент, который анализирует текст и определяет подробную информацию о шутке.
Вот список существующих категорий шуток:
{categories_str}
Выбери категорию из списка выше или создай новую категорию, если ни одна не подходит.
Ответь JSON с полями:
{{
    "category": "Категория шутки или 'Новая категория', если не подходит ни одна",
    "characters": ["список действующих лиц"],
    "tags": ["список тегов"],
    "title": "Краткое описывающее название шутки (5-10 слов)"
}}

Текст для анализа:
{{text}}
"""
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    llm = OpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    output_parser = StructuredOutputParser(pydantic_object=JokeInfoParser)
    return Node(chain=chain, output_parser=output_parser)

def get_joke_info(message_text: str) -> dict:
    init_db()
    existing_categories = get_all_categories()

    graph = LangGraph()

    is_joke_node = create_is_joke_node()
    joke_info_node = create_joke_info_node(existing_categories)

    graph.add_node("is_joke", is_joke_node)
    graph.add_node("joke_info", joke_info_node)

    graph.add_edge("is_joke", "joke_info", condition=lambda output: output.get("is_joke") == True)

    # Запускаем граф
    outputs = graph.run({"is_joke": {"text": message_text}, "joke_info": {"text": message_text}})

    is_joke_result = outputs.get("is_joke", {})
    if not is_joke_result.get("is_joke"):
        return {"is_joke": False}

    joke_info_result = outputs.get("joke_info", {})
    category_name = joke_info_result.get("category", "Новая категория")
    category_name = get_or_create_category(category_name)
    joke_info_result["category"] = category_name
    joke_info_result["is_joke"] = True

    return joke_info_result
