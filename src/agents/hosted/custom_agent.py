from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List

class CustomAgent:
    def __init__(self, model_path, device='cuda:0', output_logits=True, use_cache=True, output_attentions=True, output_scores=False, output_hidden_states=False):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.output_logits=output_logits 
        self.use_cache=use_cache
        self.output_attentions= output_attentions
        self.output_hidden_states = output_hidden_states
        self.output_scores = output_scores
    
    def generate(self, user_prompt: str, system_prompt: str, gen_strategy: Dict, assistant_prompt: str = None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user","content": user_prompt}]

        if assistant_prompt is not None:
            messages.insert(2, {"role": "assistant", "content": assistant_prompt})

        prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True if assistant_prompt is None else False
        )

        inputs = self.tokenizer(
            prompt, return_tensors='pt',
            padding=False, add_special_tokens=False)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        output_sequences = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, eos_token_id=[self.tokenizer.eos_token_id], 
            pad_token_id=self.tokenizer.eos_token_id, 
            output_logits=self.output_logits, use_cache=self.use_cache, output_attentions=self.output_attentions, return_dict_in_generate=True,
            output_hidden_states=self.output_hidden_states, output_scores=self.output_scores, **gen_strategy)

        #print(len(inputs['input_ids'][0]))
        #print(len(output_sequences['sequences'][0]))
        
        generated_text = self.tokenizer.decode(output_sequences['sequences'][0][len(input_ids[0]):], skip_special_tokens=True)
        
        return generated_text, output_sequences, inputs