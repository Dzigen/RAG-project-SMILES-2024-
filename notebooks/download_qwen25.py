from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

SAVE_MODEL_PATH = '../models/Qwen/Qwen2.5-7B-Instruct'
LOAD_MODEL = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(LOAD_MODEL)
model = AutoModelForCausalLM.from_pretrained(LOAD_MODEL)

tokenizer.save_pretrained(SAVE_MODEL_PATH)
model.save_pretrained(SAVE_MODEL_PATH)