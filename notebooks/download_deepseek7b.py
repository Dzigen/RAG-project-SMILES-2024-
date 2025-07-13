from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import  pipeline
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

SAVE_MODEL_PATH = "../models/deepseek-ai/deepseek-llm-7b-chat" 
LOAD_MODEL = "deepseek-ai/deepseek-llm-7b-chat"  

<<<<<<< HEAD
access_token = "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"
=======
access_token = "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"
>>>>>>> 74386f9 (asd)

tokenizer = AutoTokenizer.from_pretrained(LOAD_MODEL, token=access_token)
model = AutoModelForCausalLM.from_pretrained(LOAD_MODEL, torch_dtype=torch.bfloat16, token=access_token).to('cpu')

model.save_pretrained(SAVE_MODEL_PATH)
tokenizer.save_pretrained(SAVE_MODEL_PATH)
