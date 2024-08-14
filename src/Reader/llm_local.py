from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import  pipeline
from dataclasses import dataclass, field, asdict
from torch.utils.data import Dataset
import torch
import gc

from typing import List
from dataclasses import asdict
from .utils import LLM_Config
from tqdm import tqdm

class ReaderInputDataset(Dataset):
    def __init__(self, system_prompt: str, template_pipe_func, 
                 user_prompts: List[str], contexts: List[str] = None):
        self.user_prompts = user_prompts
        self.contexts = contexts
        self.template_pipe_func = template_pipe_func
        self.system_prompt = system_prompt
    
    def __len__(self):
        return len(self.user_prompts)

    def __getitem__(self, i):
        msgs = [{"role": "system", "content": self.system_prompt},
                {"role": "user","content": self.user_prompts[i]}]

        if self.contexts is not None:
            msgs.insert(1, {"role": "assistant", "content": self.contexts[i]})

        prompt = self.template_pipe_func(msgs, tokenize=False, add_generation_prompt=True)   
        
        return prompt

class LLM_Model:
    """
    Класс для инициализации модели и обращения к ней
    """

    def __init__(self, config: LLM_Config) -> None:
        """
        Инициализация модели.
        """
        
        self.config = config
        
        self.model = AutoModelForCausalLM.from_pretrained(**asdict(self.config.init))
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.init.pretrained_model_name_or_path)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, 
                             num_workers=self.config.data_operate.num_workers)

    def post_proc(self, running_pipe):
        for out in running_pipe:
            proc_out = [item['generated_text'].strip() for item in out]
            yield proc_out
    
    def generate(self, user_prompts: List[str], contexts: List[str] = None, system_prompt: str = None, 
                 gen_params: dict = None, batch_size: int = None):
        dataset = ReaderInputDataset(self.config.prompts.system if system_prompt is None else system_prompt, 
                                     self.pipe.tokenizer.apply_chat_template, 
                                     user_prompts, contexts)
        return self.post_proc(self.pipe(
            dataset, return_full_text=False, batch_size=self.config.data_operate.batch_size if batch_size is None else batch_size,
            **self.config.gen if gen_params is None else gen_params))
    
    def single_generate(self, user_prompt:str, context: str = None) -> str:
        """
        Метод для создания чата с моделью и генерации ответа на вопрос
        """

        msgs = [
            {"role": "system", "content": self.config.prompts.system},
            {"role": "user","content": user_prompt}
            ]

        if context is not None:
            msgs.insert(1, {"role": "assistant", "content": context})

        prompt = self.pipe.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)   
        outputs = self.pipe(prompt, **self.config.gen)

        return outputs[0]['generated_text'][len(prompt):].strip()

if __name__=="__main__":
    pass