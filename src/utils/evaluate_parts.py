from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List

from .evaluation_metrics import RetrieverMetrics, ReaderMetrics
from src.Retriever import ThresholdRetriever
from src.Reader import LLM_Model
from typing import Dict

def evaluate_reader(benchmarks_df: dict, reader: LLM_Model, 
                    metrics: ReaderMetrics, contexts: dict = None, show_step: int = 1, 
                    cache_relevant_flags: Dict[str, List[bool]] = None):
    scores = {}
    cache = {}
    for _, name in enumerate(benchmarks_df.keys()):
        
        scores[name] = {
            'BLEU2': [], 'BLEU1': [],
            'ExactMatch': [],'METEOR': [],
            'BertScore': [],
            'Levenshtain': [],
            'StubScore': [] # отношение числа успешно сгенерированных заглушек к их ожидаемому числу
            }
        cache[name] = []
        
        process = tqdm(reader.generate(
            benchmarks_df[name]['question'].to_list(), contexts[name] if contexts is not None else None))

        tmp_target_answers = []
        for i, predicted_answer in enumerate(process):
            print("answer raw len: ", i, len(predicted_answer))
            
            predicted_answer = predicted_answer[0]
            target_answer = benchmarks_df[name]['answer'][i]

            if cache_relevant_flags is None or cache_relevant_flags[name][i]:
                scores[name]['BLEU1'] += metrics.bleu1([predicted_answer], [target_answer])
                scores[name]['BLEU2'] += metrics.bleu2([predicted_answer], [target_answer])
                scores[name]['ExactMatch'] += metrics.exact_match([predicted_answer], [target_answer])
                scores[name]['METEOR'] += metrics.meteor([predicted_answer], [target_answer])
                scores[name]['Levenshtain'] += metrics.levenshtain_score([predicted_answer], [target_answer])
                
                tmp_target_answers.append(target_answer)
                cache[name].append(predicted_answer)
            else:
                tmp_target_answers.append(reader.config.prompts.stub_answer)
                    
                if len(contexts[name][i]) == 0:
                    scores[name]['StubScore'].append(1)
                    cache[name].append(reader.config.prompts.stub_answer)
                    
                else:
                    scores[name]['StubScore'] +=  metrics.llm_sim_score([reader.config.prompts.stub_answer], [predicted_answer])
                    cache[name].append(predicted_answer)

            print(cache_relevant_flags[name][i] if cache_relevant_flags is not None else None)
            print(predicted_answer)
            print(target_answer)
            print(scores[name]['StubScore'])
            
            if i % show_step == 0:
                process.set_postfix({m_name: np.mean(score) for m_name, score in scores[name].items()})

        scores[name] = {m_name: round(float(np.mean(score)), 5) for m_name, score in scores[name].items()}
        scores[name]['BertScore'] = metrics.bertscore(cache[name], tmp_target_answers)
        scores[name]['elapsed_time_sec'] = round(float(process.format_dict["elapsed"]), 3)
        process.set_postfix(scores[name])

    return scores, cache


def evaluate_retriever(benchmarks_df: dict, retrievers: dict, metrics: RetrieverMetrics, 
                       show_step: int = 50, topk_score_list: int = 3):
    scores = {}
    cache_ids = {}
    cache_docs = {}
    cache_relevant_flags = {}
    for _, name in enumerate(benchmarks_df.keys()):
        print(name)

        scores[name] = {
            'MRR': [], 'mAP': [],
            'Recall': [], 'Precision': [],
            'F1': [],
            'NoRelContextScore': []}
        cache_ids[name] = []
        cache_docs[name] = []
        cache_relevant_flags[name] = []
        
        process = tqdm(range(benchmarks_df[name].shape[0])) 
        for i in process:
            predicted_chunks = retrievers[name].invoke(benchmarks_df[name]['question'][i])
            predicted_chunk_ids = list(map(lambda item: item[2]['chunk_id'], predicted_chunks))
            target_chunk_ids = benchmarks_df[name]['chunk_ids'][i]
            
            #print(len(predicted_chunk_ids), target_chunk_ids[0] in predicted_chunk_ids)
            
            scores[name]['MRR'].append(metrics.reciprocal_rank(predicted_chunk_ids, target_chunk_ids))
            scores[name]['mAP'].append(metrics.AP(predicted_chunk_ids, target_chunk_ids))
            scores[name]['Recall'].append(metrics.recall(predicted_chunk_ids, target_chunk_ids, k=topk_score_list))
            scores[name]['Precision'].append(metrics.precision(predicted_chunk_ids, target_chunk_ids, k=topk_score_list))
                        
            f1_score = metrics.f1_score(predicted_chunk_ids, target_chunk_ids, k=topk_score_list)
            scores[name]['F1'].append(0 if np.isnan(f1_score) else f1_score)

            cache_ids[name].append(predicted_chunk_ids)
            cache_docs[name].append(predicted_chunks)
            #print(target_chunk_ids)
            cache_relevant_flags[name].append(target_chunk_ids[0] in predicted_chunk_ids)

            if i % show_step == 0:
                process.set_postfix({m_name: np.mean(score) for m_name, score in scores[name].items()})

        scores[name] = {m_name: round(float(np.mean(score)), 5) for m_name, score in scores[name].items()}
        scores[name]['NoRelContextScore'] = sum(cache_relevant_flags[name])
        scores[name]['elapsed_time_sec'] = round(float(process.format_dict["elapsed"]), 3)
        process.set_postfix(scores[name])
        
    return scores, cache_ids, cache_docs, cache_relevant_flags