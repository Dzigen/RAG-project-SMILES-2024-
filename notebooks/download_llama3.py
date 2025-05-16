from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import  pipeline
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

SAVE_MODEL_PATH = '../models/Undi95/Meta-Llama-3-8B-Instruct-hf'
LOAD_MODEL = "Undi95/Meta-Llama-3-8B-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(LOAD_MODEL)
model = AutoModelForCausalLM.from_pretrained(LOAD_MODEL)

tokenizer.save_pretrained(SAVE_MODEL_PATH)
model.save_pretrained(SAVE_MODEL_PATH)