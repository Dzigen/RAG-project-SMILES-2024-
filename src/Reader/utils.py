from dataclasses import dataclass, field
import torch

@dataclass
class LLM_PromptsConfig:
    assistant: str = "Отвечай на вопросы, используя информацию из текстов в списке ниже:"
    system: str = "Ты вопросно-ответная система. Все ответы генерируй на русском языке. По вопросам отвечай чётко и конкретно."
    stub_answer: str = "У меня нет ответа на ваш вопрос."

@dataclass
class LLM_InitConfig:
    pretrained_model_name_or_path: str = '/trinity/home/team06/workspace/mikhail_workspace/rag_project/models/vikhr_7B'
    device_map: str = "cuda"
    torch_dtype: float = torch.bfloat16
    attn_implementation: str = "flash_attention_2"

@dataclass
class LLM_DataOperateConfig:
    batch_size: int = 1
    num_workers: int = 8

@dataclass
class LLM_Config:
    prompts: LLM_PromptsConfig = LLM_PromptsConfig() 
    init: LLM_InitConfig = LLM_InitConfig()
    data_operate: LLM_DataOperateConfig = LLM_DataOperateConfig() 
    gen: dict = field(default_factory=lambda: {'max_new_tokens': 512, 'num_beams': 5, 'eos_token_id': 79097, 'early_stopping': True})
    

