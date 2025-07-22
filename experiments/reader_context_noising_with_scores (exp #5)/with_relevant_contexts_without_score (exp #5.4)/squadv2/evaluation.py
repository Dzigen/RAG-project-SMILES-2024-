import sys
BASE_DIR = "../../../.."
sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
import ast
import random
import json
from time import time
import gc
import os
import joblib
from copy import deepcopy
import torch
import chromadb
import gc
from tqdm import tqdm
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from dataclasses import dataclass
import matplotlib.pyplot as plt

random.seed(42)

from src.agents.hosted import CustomAgent
from src.utils import ReaderMetrics
from src.utils.inference_metrics import compute_predictive_entropy, get_timportance_info

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

CONTEXTS_DATASET_PATH = "../../../../data/squadv2/contexts.csv"
QA_DATASET_PATH = "../../../../data/squadv2/qa_dataset.csv"
#AGENT_MODEL_PATH = "../../../../models/Qwen/Qwen2.5-7B-Instruct" # "Undi95/Meta-Llama-3-8B-Instruct-hf" / "../../../../models/Qwen/Qwen2.5-7B-Instruct"
AGENT_MODEL_PATH = "Undi95/Meta-Llama-3-8B-Instruct-hf"

########################################

PARAMS = {
    'version': "1.1.1.attn_inptonly",
    'num_samples': 2000,
    'num_contexts': 5,
    'model': AGENT_MODEL_PATH,
    'system_prompt': "You are an AI assistant who helps solve user issues.",
    "item_format": "- {document}",
    "user_prompt": 'Answer the question using the available information from the texts in the list below. If there are no texts in the list that are relevant enough to generate answer based on them, then generate the following text: "I do not have an answer to your question". Generate answer only in English. Do not duplicate the question in the answer. Generate only the answer to the specified question. Answer need to be short. Do not generate anything extra.',
    "prompt_format": "{user_p}\n\nAvailable information:\n{cnt_list}\n\nQuestion:\n{q}\n\nAnswer:\n",
    'scores': {'rel': 1.0, 'unrel': 0.0},
    'gen_strat': {'max_new_tokens': 1024, 'do_sample': False, 'num_beams': 1},
    'stub_answer': "I do not have an answer to your question",
    'calculate_entropy': True,
    'calculate_timportance(attention)': True,
    'timportnace_hyperp': {
        'only_for_input_tokens': True, 
        #'layers': [0,1,13,27], # qwen2.5
        'layers': [0,1,15,31], # llama3.1
        'mean_by': ['columns', 'rows']},
    'revert': False,
    'centered': False
}

METADATA_SAVE_NAME = 'metadata.json'
USER_PROPMTS_SAVE_NAME = 'user_prompts.json'
PARAMS_SAVE_NAME = 'hyperp.json'
GEN_ANSW_SAVE_NAME = 'generation_info.json'
SCORES_SAVE_NAME = 'scores.json'
TIMPORTANCE_SAVE_NAME = 'timportance'
LOGS_SAVE_DIR = './logs_v2'
META_INFO_DIR_NAME = 'gen_metainfo'

if os.path.exists(f'{LOGS_SAVE_DIR}/v{PARAMS["version"]}'):
    print("Dir exists")
else:
    print("Creating Dir...")
    os.mkdir(f'{LOGS_SAVE_DIR}/v{PARAMS["version"]}')
    os.mkdir(f'{LOGS_SAVE_DIR}/v{PARAMS["version"]}/{META_INFO_DIR_NAME}')

########################################

agent = CustomAgent(PARAMS['model'], output_logits=PARAMS['calculate_entropy'], use_cache=True, output_attentions=False, output_scores=False, output_hidden_states=False)
output = agent.generate(user_prompt="what is wrong with humanity?", system_prompt=PARAMS['system_prompt'], 
                        gen_strategy=PARAMS['gen_strat'])
print(output[0])

########################################

agent.model

########################################

dataset_df = pd.read_csv(QA_DATASET_PATH)
contexts_df = pd.read_csv(CONTEXTS_DATASET_PATH)

CONTEXTS_LIST_IDS = []
for i in tqdm(range(PARAMS['num_samples'])):
    cur_rel_id = int(dataset_df['relevant_context_id'][i])
    cur_list_ids = [(-1, cur_rel_id)]

    while len(cur_list_ids) != PARAMS['num_contexts']:
        unrel_context_id = random.randint(0, contexts_df.shape[0]-1)

        prep_cntx = (-1, unrel_context_id)
        if unrel_context_id != cur_rel_id:
            cur_list_ids.append(prep_cntx)

    # shuffling strategy
    if PARAMS['revert']:
        cur_list_ids = cur_list_ids[::-1]
    elif PARAMS['centered']:
        cur_list_ids.pop(0)
        cur_list_ids.insert(len(cur_list_ids)//2, (-1, cur_rel_id))
    
    CONTEXTS_LIST_IDS.append(cur_list_ids)

print(CONTEXTS_LIST_IDS[0])

########################################

USER_PROMPTS = []
gc.collect()
for i in tqdm(range(len(CONTEXTS_LIST_IDS))):
    docs = [contexts_df['context'][CONTEXTS_LIST_IDS[i][j][1]] for j in range(len(CONTEXTS_LIST_IDS[i]))]
    documents_list = [PARAMS['item_format'].format(document=doc.strip()) for doc in docs]
    
    documents_list = '\n'.join(documents_list)
    USER_PROMPTS.append(PARAMS['prompt_format'].format(user_p=PARAMS['user_prompt'], cnt_list=documents_list, q=dataset_df['question'][i]))

########################################

with open(f"{LOGS_SAVE_DIR}/v{PARAMS['version']}/{USER_PROPMTS_SAVE_NAME}", 'w', encoding='utf-8') as fp:
    fp.write(json.dumps(USER_PROMPTS, ensure_ascii=False, indent=1))

# сохраняем конфигурацию эксперимента
with open(f"{LOGS_SAVE_DIR}/v{PARAMS['version']}/{PARAMS_SAVE_NAME}", 'w', encoding='utf-8') as fp:
    fp.write(json.dumps(PARAMS, ensure_ascii=False, indent=1))

del contexts_df
gc.collect()
print(USER_PROMPTS[0])

########################################

generate_answers, calc_metrics, input_prompts = [], [], []
timportance_info = dict()
display_iter = 100
s_time = time()
for i in tqdm(range(len(USER_PROMPTS))):
    if PARAMS['calculate_timportance(attention)'] and PARAMS['timportnace_hyperp']['only_for_input_tokens']:
        agent.output_attentions = True
        _, meta_info, inputs, cur_prompt = agent.generate(
            user_prompt=USER_PROMPTS[i], system_prompt=PARAMS['system_prompt'], 
            gen_strategy={'max_new_tokens': 1, 'do_sample': False, 'num_beams': 1})
        
        timportance_info[i] = get_timportance_info(
            meta_info['attentions'][0], layer_ids = PARAMS['timportnace_hyperp']['layers'], 
            mean_attn = PARAMS['timportnace_hyperp']['mean_by'])
        agent.output_attentions = False
    else:
        pred_answer, meta_info, inputs, cur_prompt = agent.generate(
            user_prompt=USER_PROMPTS[i], system_prompt=PARAMS['system_prompt'],
            gen_strategy=PARAMS['gen_strat'])


    cur_metrics = dict()
    cur_metrics['input_tokens'] = inputs['input_ids'].shape[1]
    cur_metrics['gen_tokens'] = meta_info['sequences'].shape[1] - cur_metrics['input_tokens']
    
    if PARAMS['calculate_entropy']:
        logits = torch.cat(meta_info['logits'], 0).cpu().detach()
        entropy = compute_predictive_entropy(logits)
        cur_metrics['predictive_entropy'] = float(entropy)
    
    pred_answer = None
    if PARAMS['calculate_timportance(attention)'] and not PARAMS['timportnace_hyperp']['only_for_input_tokens']:
        agent.output_attentions = True
        tmp_assistant_prompt = pred_answer
        _, meta_info, _, _ = agent.generate(
            user_prompt=USER_PROMPTS[i], system_prompt=PARAMS['system_prompt'], 
            gen_strategy={'max_new_tokens': 1, 'do_sample': False, 'num_beams': 1}, 
            assistant_prompt=tmp_assistant_prompt)
        
        timportance_info[i] = get_timportance_info(
            meta_info['attentions'][0], layer_ids = PARAMS['timportnace_hyperp']['layers'], 
            mean_attn = PARAMS['timportnace_hyperp']['mean_by'])
        agent.output_attentions = False
        
    calc_metrics.append(cur_metrics)
    generate_answers.append(pred_answer)
    input_prompts.append(cur_prompt)
    
    # logits = torch.cat(meta_info['logits'], 0).cpu().detach().numpy()
    # logits_int8 = logits.astype('int8') 
    # token_logits = {f"token_{i}": token_logits for i, token_logits in enumerate(logits_int8)}
    # pa_table = pa.table(token_logits)
    # pa.parquet.write_table(pa_table, f"{LOGS_SAVE_DIR}/v{PARAMS['version']}/{META_INFO_DIR_NAME}/logits_{i}.parquet")
    
    if i % display_iter == 0:
        print(f"\n[{i}]: \nGEN: {pred_answer}\nGOLD: {dataset_df['answer'][i]}\nMETRICS: {cur_metrics}")
e_time = time()


# сохраняем используемые контексты + сгнерированные ответы
gen_info = []
for i in range(PARAMS['num_samples']):
    formated_contexts = [(float(item[0]), int(item[1])) for item in CONTEXTS_LIST_IDS[i]]
    cur_item = {
        'input_prompt': input_prompts[i],
        'gen_answer': str(generate_answers[i]), 
        'metainfo': calc_metrics[i], 
        'used_contexts': formated_contexts}
    gen_info.append(cur_item)

with open(f"{LOGS_SAVE_DIR}/v{PARAMS['version']}/{GEN_ANSW_SAVE_NAME}", 'w', encoding='utf-8') as fp:
    fp.write(json.dumps(gen_info, ensure_ascii=False, indent=1))

with open(f"{LOGS_SAVE_DIR}/v{PARAMS['version']}/{METADATA_SAVE_NAME}", 'w', encoding='utf-8') as fp:
    fp.write(json.dumps({'elapsed_time': e_time - s_time}, ensure_ascii=False, indent=1))

if PARAMS['calculate_timportance(attention)']:
    joblib.dump(timportance_info, f"{LOGS_SAVE_DIR}/v{PARAMS['version']}/{META_INFO_DIR_NAME}/{TIMPORTANCE_SAVE_NAME}")

########################################

LOADING_VERSION = PARAMS['version']

with open(f'{LOGS_SAVE_DIR}/v{LOADING_VERSION}/{GEN_ANSW_SAVE_NAME}','r', encoding='utf8') as fd:
    predicted_answers = list(map(lambda v: v['gen_answer'], json.loads(fd.read())))

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

metrics = ReaderMetrics(base_dir=BASE_DIR, model_path='en_electra_base')
dataset_df = pd.read_csv(QA_DATASET_PATH)

target_scores = {
    'BLEU2': [], 'BLEU1': [],
    'ExactMatch': [],'METEOR': [],
    'BertScore': [],
    'Levenshtain': [],
    'ROUGEL': []}

stub_scores = {
    'BLEU2': [], 'BLEU1': [],
    'ExactMatch': [],'METEOR': [],
    'BertScore': [],
    'Levenshtain': [],
    'ROUGEL': []}

show_step = 10

process = tqdm(range(PARAMS['num_samples']))
target_answers =  dataset_df['answer'].to_list()[:PARAMS['num_samples']]
tmp_stub_pred_answers = []
for i in process:
    
    predicted_answer = predicted_answers[i]
    target_answer = target_answers[i]

    target_scores['BLEU1'] += metrics.bleu1([predicted_answer], [target_answer])
    target_scores['BLEU2'] += metrics.bleu2([predicted_answer], [target_answer])
    target_scores['ExactMatch'] += metrics.exact_match([predicted_answer], [target_answer])
    target_scores['METEOR'] += metrics.meteor([predicted_answer], [target_answer])
    target_scores['Levenshtain'] += metrics.levenshtain_score([predicted_answer], [target_answer])
    target_scores['ROUGEL'] += metrics.rougel([predicted_answer], [target_answer])


    stub_pred_answer = predicted_answer
    tmp_stub_pred_answers.append(stub_pred_answer)

    stub_scores['BLEU1'] += metrics.bleu1([stub_pred_answer], [PARAMS['stub_answer']])
    stub_scores['BLEU2'] += metrics.bleu2([stub_pred_answer], [PARAMS['stub_answer']])
    stub_scores['ExactMatch'] += metrics.exact_match([stub_pred_answer], [PARAMS['stub_answer']])
    stub_scores['METEOR'] += metrics.meteor([stub_pred_answer], [PARAMS['stub_answer']])
    stub_scores['Levenshtain'] += metrics.levenshtain_score([stub_pred_answer], [PARAMS['stub_answer']])
    stub_scores['ROUGEL'] += metrics.rougel([stub_pred_answer], [PARAMS['stub_answer']])
            
    if i % show_step == 0:
        process.set_postfix({m_name: np.mean(score) for m_name, score in stub_scores.items()})

target_scores = {m_name: round(float(np.mean(score)), 5) for m_name, score in target_scores.items()}
#target_scores['BertScore'] = metrics.bertscore(predicted_answers, target_answers)

stub_scores = {m_name: round(float(np.mean(score)), 5) for m_name, score in stub_scores.items()}
#stub_scores['BertScore'] = metrics.bertscore(tmp_stub_pred_answers, [PARAMS['stub_answer']]*len(tmp_stub_pred_answers))
stub_scores['elapsed_time_sec'] = round(float(process.format_dict["elapsed"]), 3)

########################################

with open(f"{LOGS_SAVE_DIR}/v{LOADING_VERSION}/{SCORES_SAVE_NAME}", 'w', encoding='utf-8') as fp:
    fp.write(json.dumps({'target_answers': target_scores, 'stub_answers': stub_scores}, ensure_ascii=False, indent=1))
