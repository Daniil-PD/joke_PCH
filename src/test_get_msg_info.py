from get_msg_info import get_joke_info

def test_get_joke_info():
    test_messages = [
        "Почему программисты не любят природу? Потому что там слишком много багов.",
        "Это просто обычное сообщение без шуток.",
        "Анекдот: Встречаются два друга, один говорит другому: 'Ты слышал последнюю шутку про базу данных?'",
    ]

    for msg in test_messages:
        result = get_joke_info(msg)
        print(f"Message: {msg}")
        print(f"Result: {result}")
        print("-" * 40)

if __name__ == "__main__":
    test_get_joke_info()