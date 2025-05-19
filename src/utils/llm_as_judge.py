from mistralai import Mistral
import re

api_key = ... # TO FILL
model = "mistral-large-latest"

RU_USER_PROMPT = """Тебе нужно оценить качество ответа на вопрос. Тебе даны: вопрос, ground truth ответ, который точно правильный, и сгенерированный ответ, которому ты должна поставить оценку  1 или 0. Оценка 1 означает, что ответ правильный, оценка 0 означает, что ответ неправильный. Нужно сравнить сгенерированный ответ с ground truth ответом, принимая во внимание и сам вопрос. Ground truth ответ и сгенерированный ответ могут быть разной длинны, содержать разное количество информации, поэтому учитывай то, дан ли ответ на вопросительное слово из вопроса: если в вопросе "кто", то в сгенерированном ответе должен быть назван какой-то человек, персонаж или животное; если в вопросе "сколько", то в ответе должно быть число.

Пример 1:
Вопрос: "На какое место Everybody поднимается в чарте Hot Dance Club Songs?"
Ground truth ответ: "На 3-е место Everybody поднимается в чарте Hot Dance Club Songs."
Сгенерированный ответ: "3-е место."
Оценка модели: 1

Пример 2:
Вопрос: "В какой области много многолюдных поселков?"
Ground truth ответ: "В области земель Колхиды."
Сгенерированный ответ: "В этой области много многолюдных поселков."
Оценка модели: 0

Пример 3:
Вопрос: "В скольких томах опубликована История Рима?"
Ground truth ответ: "История Рима опубликована в четырёх томах."
Сгенерированный ответ: "Ответ: В трех томах."
Оценка модели: 0

Пример 4:
Вопрос: "Какая голова была у Микеланджело?"
Ground truth ответ: "У Микеланджело была круглая голова."
Сгенерированный ответ: "Голова у Микеланджело была круглая, лоб квадратный, изрезанный морщинами, с сильно выраженными надбровными дугами."
Оценка модели: 1


Вопрос: "{q}"
Ground truth ответ: "{gt_a}"
Сгенерированный ответ: "{g_a}"
Оценка модели:
"""

EN_USER_PROMPT = """"""

def parse_score(raw_output):
    return int(raw_output.split("\n")[0].split(' ')[-1])

def call_mistral(
    api_key: str,
    model_name: str = "mistral-large-latest",
    system_prompt: str = None,
    user_prompt: str = None,
    assistant_prompt: str = None,
    gen_config: dict = None
) -> str:

    
    client = Mistral(api_key=api_key)
    messages = []
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
    if assistant_prompt is not None:
        messages.append({
            "role": "assistant",
            "content": assistant_prompt
        })
        
    if user_prompt is not None:
        messages.append({
            "role": "user",
            "content": user_prompt
        });

    if gen_config is None:
        gen_config = {}
        
    chat_response = client.chat.complete(
        model=model_name,
        messages=messages,
        **gen_config
    )

    return chat_response.choices[0].message.content