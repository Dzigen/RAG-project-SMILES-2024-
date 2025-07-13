import sys
BASE_DIR = "/home/jovyan/work/alexander_workspace/RAG-project-SMILES-2024-"
sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
import ast
import random
import json
from time import time
import gc
import os
import torch
import joblib
from tqdm import tqdm
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from dataclasses import dataclass
import pyarrow as pa

random.seed(42)

from src.agents.hosted import CustomAgent
from src.utils import ReaderMetrics
from src.utils.inference_metrics import compute_predictive_entropy, get_timportance_info

###########################

print("Loading configuration...")

import yaml
PARAMS_FILE = sys.argv[1]
with open(PARAMS_FILE) as stream:
    PARAMS = yaml.safe_load(stream)

BAGPACK_FILE = 'bagpack.json'
with open(BAGPACK_FILE, 'r', encoding='utf-8') as fd:
    BAGPACK = json.loads(fd.read())
    
METADATA_SAVE_NAME = 'metadata.json'
USER_PROPMTS_SAVE_NAME = 'user_prompts.json'
PARAMS_SAVE_NAME = 'hyperp.json'
GEN_ANSW_SAVE_NAME = 'generation_info.json'
SCORES_SAVE_NAME = 'scores.json'
TIMPORTANCE_SAVE_NAME = 'timportance'
META_INFO_DIR_NAME = 'gen_metainfo'

if os.path.exists(f'{PARAMS["LOGS_SAVE_DIR"]}/v{PARAMS["version"]}'):
    print("Dir exists")
else:
    print("Creating Dir...")
    os.mkdir(f'{PARAMS["LOGS_SAVE_DIR"]}/v{PARAMS["version"]}')
    os.mkdir(f'{PARAMS["LOGS_SAVE_DIR"]}/v{PARAMS["version"]}/{META_INFO_DIR_NAME}')

###########################

print("Loading model...")

agent = CustomAgent(
    PARAMS['model'], output_logits=PARAMS['calculate_entropy'], use_cache=True, 
    output_attentions=False, output_scores=False, output_hidden_states=False)

###########################

dataset_df = pd.read_csv(PARAMS['QA_DATASET_PATH'])
contexts_df = pd.read_csv(PARAMS['CONTEXTS_DATASET_PATH'])

print("Preparing contexts-sequence...")

CONTEXTS_LIST_IDS = []
for i in tqdm(range(PARAMS['num_samples'])):
    cur_rel_id = dataset_df['relevant_context_id'][i]
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

###########################

print("Preparing user-prompt...")

USER_PROMPTS = []
gc.collect()
for i in tqdm(range(len(CONTEXTS_LIST_IDS))):
    docs = [contexts_df['context'][CONTEXTS_LIST_IDS[i][j][1]] for j in range(len(CONTEXTS_LIST_IDS[i]))]
    documents_list = [BAGPACK[PARAMS['bp']]['item_format'].format(document=doc.strip()) for doc in docs]
    
    documents_list = '\n'.join(documents_list)
    USER_PROMPTS.append(BAGPACK[PARAMS['bp']]['prompt_format'].format(user_p=BAGPACK[PARAMS['bp']]['user_prompt'], cnt_list=documents_list, q=dataset_df['question'][i]))

with open(f"{PARAMS['LOGS_SAVE_DIR']}/v{PARAMS['version']}/{USER_PROPMTS_SAVE_NAME}", 'w', encoding='utf-8') as fp:
    fp.write(json.dumps(USER_PROMPTS, ensure_ascii=False, indent=1))

# сохраняем конфигурацию эксперимента
with open(f"{PARAMS['LOGS_SAVE_DIR']}/v{PARAMS['version']}/{PARAMS_SAVE_NAME}", 'w', encoding='utf-8') as fp:
    fp.write(json.dumps(PARAMS, ensure_ascii=False, indent=1))

print(USER_PROMPTS[0])

del contexts_df
gc.collect()

###########################

print("Generating answers...")

generate_answers, calc_metrics = [], []
timportance_info = dict()
display_iter = 100
s_time = time()
for i in tqdm(range(len(USER_PROMPTS))):
    pred_answer, meta_info, inputs = agent.generate(
        user_prompt=USER_PROMPTS[i], system_prompt=BAGPACK[PARAMS['bp']]['system_prompt'],
        gen_strategy=PARAMS['gen_strat'])

    cur_metrics = dict()
    cur_metrics['input_tokens'] = inputs['input_ids'].shape[1]
    cur_metrics['gen_tokens'] = meta_info['sequences'].shape[1] - cur_metrics['input_tokens']
    
    if PARAMS['calculate_entropy']:
        logits = torch.cat(meta_info['logits'], 0).cpu().detach()
        entropy = compute_predictive_entropy(logits)
        cur_metrics['predictive_entropy'] = float(entropy)
    
    if PARAMS['calculate_timportance(attention)']:
        agent.output_attentions = True
        tmp_assistant_prompt = pred_answer
        _, meta_info, _ = agent.generate(
            user_prompt=USER_PROMPTS[i], system_prompt=BAGPACK[PARAMS['bp']]['system_prompt'], 
            gen_strategy={'max_new_tokens': 1, 'do_sample': False, 'num_beams': 1}, 
            assistant_prompt=tmp_assistant_prompt)
        timportance_info[i] = get_timportance_info(
            meta_info['attentions'][0], layer_ids = PARAMS['timportnace_hyperp']['layers'], 
            mean_attn = PARAMS['timportnace_hyperp']['mean_by'])
        agent.output_attentions = False
        
    calc_metrics.append(cur_metrics)
    generate_answers.append(pred_answer)
    
    # logits = torch.cat(meta_info['logits'], 0).cpu().detach().numpy()
    # logits_int8 = logits.astype('int8') 
    # token_logits = {f"token_{i}": token_logits for i, token_logits in enumerate(logits_int8)}
    # pa_table = pa.table(token_logits)
    # pa.parquet.write_table(pa_table, f"{LOGS_SAVE_DIR}/v{PARAMS['version']}/{META_INFO_DIR_NAME}/logits_{i}.parquet")
    
    if i % display_iter == 0:
        print(f"\n[{i}]: \nGEN: {pred_answer}\nGOLD: {dataset_df['answer'][i]}\nMETRICS: {cur_metrics}")
e_time = time()

###########################

print("Saving generation results...")

# сохраняем используемые контексты + сгнерированные ответы
gen_info = []
for i in range(PARAMS['num_samples']):
    formated_contexts = [(float(item[0]), int(item[1])) for item in CONTEXTS_LIST_IDS[i]]
    cur_item = {
        'gen_answer': str(generate_answers[i]), 
        'metainfo': calc_metrics[i], 
        'used_contexts': formated_contexts}
    gen_info.append(cur_item)

with open(f"{PARAMS['LOGS_SAVE_DIR']}/v{PARAMS['version']}/{GEN_ANSW_SAVE_NAME}", 'w', encoding='utf-8') as fp:
    fp.write(json.dumps(gen_info, ensure_ascii=False, indent=1))

with open(f"{PARAMS['LOGS_SAVE_DIR']}/v{PARAMS['version']}/{METADATA_SAVE_NAME}", 'w', encoding='utf-8') as fp:
    fp.write(json.dumps({'elapsed_time': e_time - s_time}, ensure_ascii=False, indent=1))

if PARAMS['calculate_timportance(attention)']:
    joblib.dump(timportance_info, f"{PARAMS['LOGS_SAVE_DIR']}/v{PARAMS['version']}/{META_INFO_DIR_NAME}/{TIMPORTANCE_SAVE_NAME}")

###########################

print("Preparing evaluation environment...")
    
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

with open(f'{PARAMS["LOGS_SAVE_DIR"]}/v{PARAMS["version"]}/{GEN_ANSW_SAVE_NAME}','r', encoding='utf8') as fd:
    predicted_answers = list(map(lambda v: v['gen_answer'], json.loads(fd.read())))

metrics = ReaderMetrics(base_dir=BASE_DIR, model_path='en_electra_base')

dataset_df = pd.read_csv(PARAMS['QA_DATASET_PATH'])

###########################

print("Evaluating generated answers...")

target_scores = {
    'BLEU2': [], 'BLEU1': [],
    'ExactMatch': [],'METEOR': [],
    'Levenshtain': [],
    'ROUGEL': []}

stub_scores = {
    'BLEU2': [], 'BLEU1': [],
    'ExactMatch': [],'METEOR': [],
    'Levenshtain': [],
    'ROUGEL': []}

show_step = 50

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
    
    stub_scores['BLEU1'] += metrics.bleu1([stub_pred_answer], [BAGPACK[PARAMS['bp']]['stub_answer']])
    stub_scores['BLEU2'] += metrics.bleu2([stub_pred_answer], [BAGPACK[PARAMS['bp']]['stub_answer']])
    stub_scores['ExactMatch'] += metrics.exact_match([stub_pred_answer], [BAGPACK[PARAMS['bp']]['stub_answer']])
    stub_scores['METEOR'] += metrics.meteor([stub_pred_answer], [BAGPACK[PARAMS['bp']]['stub_answer']])
    stub_scores['Levenshtain'] += metrics.levenshtain_score([stub_pred_answer], [BAGPACK[PARAMS['bp']]['stub_answer']])
    stub_scores['ROUGEL'] += metrics.rougel([stub_pred_answer], [BAGPACK[PARAMS['bp']]['stub_answer']])
            
    if i % show_step == 0:
        process.set_postfix({m_name: np.mean(score) for m_name, score in stub_scores.items()})

target_scores = {m_name: round(float(np.mean(score)), 5) for m_name, score in target_scores.items()}
#target_scores['BertScore'] = metrics.bertscore(predicted_answers, target_answers)

stub_scores = {m_name: round(float(np.mean(score)), 5) for m_name, score in stub_scores.items()}
#stub_scores['BertScore'] = metrics.bertscore(tmp_stub_pred_answers, [BAGPACK[PARAMS['bp']]['stub_answer']]*len(tmp_stub_pred_answers))
stub_scores['elapsed_time_sec'] = round(float(process.format_dict["elapsed"]), 3)

###########################

print("Saving calculated scores...")

with open(f"{PARAMS['LOGS_SAVE_DIR']}/v{PARAMS['version']}/{SCORES_SAVE_NAME}", 'w', encoding='utf-8') as fp:
    fp.write(json.dumps({'target_answers': target_scores, 'stub_answers': stub_scores}, ensure_ascii=False, indent=1))