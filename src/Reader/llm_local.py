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
            n_ctx=self.conf_Hard.n_ctx,
            verbose=self.conf_Hard.verbose
        )

    def generate(self, user_prompt:str, context: str = None) -> str:

        """
        Метод для создания чата с моделью и генерации ответа на вопрос

        Параметры:
        -user_prompt:str: User промпт для LLM.

        Контент ответа:
        - output['choices'][0]["message"]['content']
        """

        msgs = [
            {"role": "system", "content": self.conf_Hard.system_prompt},
            {"role": "user","content": user_prompt}
            ]

        if context is not None:
            msgs.insert(1, {"role": "assistant", "content": context},)

        output = self.model.create_chat_completion(
            messages = msgs,
            **asdict(self.conf_Hyper)
        )

        return output

if __name__=="__main__":
    pass