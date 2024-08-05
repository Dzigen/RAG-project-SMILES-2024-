from llama_cpp import Llama
from typing import List

from dataclasses import asdict
from .utils import LLM_Hardw_Conf
from .utils import LLM_Hyper_Conf

class LLM_model:
    """
    Класс для инициализации модели и обращения к ней
    """


    def __init__(self, conf_Hard: LLM_Hardw_Conf, conf_Hyper: LLM_Hyper_Conf) -> None:

        """
        Инициализация модели.
        
        Параметры:
        -conf_Hard: Конфиг с аппаратными параметрами работы LLM; \n
        -conf_Hyper: Конфиг с гиперпараметрами модели.
        """

        self.conf_Hard = conf_Hard
        self.conf_Hyper = conf_Hyper

        self.model = Llama(
            model_path=self.conf_Hard.model_path,
            n_gpu_layers=self.conf_Hard.n_gpu_layers, 
            seed=self.conf_Hard.seed,
            n_ctx=self.conf_Hard.n_ctx
        )

    def prepare_assist_content(self, docs: List[str]) -> str:
        return '\n\n'.join(docs)

    def generate(self, user_prompt:str, docs: List[str]) -> str:

        """
        Метод для создания чата с моделью и генерации ответа на вопрос

        Параметры:
        -user_prompt:str: User промпт для LLM.

        Контент ответа:
        - output['choices'][0]["message"]['content']
        """
        
        assist_content = self.prepare_assist_content(docs)

        output = self.model.create_chat_completion(
            messages = [
            {"role": "system", "content": f"{self.conf_Hard.system_prompt}\n\n{assist_content}"},
            {"role": "assistant", "content": self.conf_Hard.assistant_prompt},
            {"role": "user","content": user_prompt}
            ],
            **asdict(self.conf_Hyper)
        )

        return output

if __name__=="__main__":

    conf1 = LLM_Hardw_Conf()
    conf2 = LLM_Hyper_Conf()

    model = LLM_model(conf1, conf2)
    
    assistant_context = "Это твоя база знаний. Используй её при ответе: Вектор – это направленный отрезок прямой, т. е. отрезок, имеющий определенную длину и определенное направление."
    user_prompt = 'Что такое вектор?'

    output = model.generate(assistant_context, user_prompt)

    for item in output:
        try:
            print(item['choices'][0]['delta']['content'], end='')
        except Exception:
            continue