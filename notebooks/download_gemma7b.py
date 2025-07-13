from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import  pipeline
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

SAVE_MODEL_PATH = "../models/google/gemma-7b-it" 
LOAD_MODEL = "google/gemma-7b-it"  

access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(LOAD_MODEL, token=access_token)
model = AutoModelForCausalLM.from_pretrained(LOAD_MODEL, torch_dtype=torch.bfloat16, token=access_token).to('cpu')

model.save_pretrained(SAVE_MODEL_PATH)
tokenizer.save_pretrained(SAVE_MODEL_PATH)