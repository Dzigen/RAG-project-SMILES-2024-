from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List

from .evaluation_metrics import RetrieverMetrics, ReaderMetrics
from src.Retriever import ThresholdRetriever
from src.Reader import LLM_model

def evaluate_reader(benchmarks_df: dict, reader: LLM_model, 
                    metrics: ReaderMetrics, contexts: List[str] = None):
    scores = {}
    cache = {}
    for _, name in enumerate(benchmarks_df.keys()):
        
        scores[name] = {
            'BLEU2': [], 'BLEU1': [],
            'ExactMatch': [],'METEOR': []
            }
        cache[name] = []

        process = tqdm(range(benchmarks_df[name].shape[0])) 
        show_step = 500
        for i in process:
            question = benchmarks_df[name]['question'][i]
            predicted_answer = reader.generate(question, None if contexts is None else contexts[i])
            target_answer = benchmarks_df[name]['question'][i]

            scores['BLEU1'].append(metrics.bleu1([predicted_answer], [target_answer]))
            scores['BLEU2'].append(metrics.bleu2([predicted_answer], [target_answer]))
            scores['ExactMatch'].append(metrics.exact_match([predicted_answer], [target_answer]))
            scores['METEOR'].append(metrics.meteor([predicted_answer], [target_answer]))

            cache[name].append(target_answer)

            if i % show_step == 0:
                process.set_postfix({name: np.median(scores) for name, scores in scores[name].items()})

        scores[name] = {name: np.median(scores) for name, scores in scores[name].items()}
        process.set_postfix(scores[name])

    return scores, cache


def evaluate_retriever(benchmarks_df: dict, retrievers_config: dict, 
                       metrics: RetrieverMetrics):

    scores = {}
    cache_ids = {}
    cache_texts = {}
    for _, name in enumerate(benchmarks_df.keys()):
        print(name)

        RETRIEVER = ThresholdRetriever(retrievers_config[name])
        TOPK_THRESHOLD = retrievers_config[name].params['max_k']

        scores[name] = {
            'MRR': [], 'mAP': [],
            'Recall': [], 'Precision': [],
            'F1': [] 
        }
        cache_ids[name] = []
        cache_texts[name] = []

        process = tqdm(range(benchmarks_df[name].shape[0])) 
        show_step = 500
        for i in process:
            predicted_chunks = RETRIEVER.invoke(benchmarks_df[name]['question'][i])
            predicted_chunk_ids = list(map(lambda item: item[2]['chunk_id'], predicted_chunks))
            target_chunk_ids = benchmarks_df[name]['chunk_ids'][i]

            scores[name]['MRR'].append(metrics.reciprocal_rank(predicted_chunk_ids, target_chunk_ids))
            scores[name]['mAP'].append(metrics.AP(predicted_chunk_ids, target_chunk_ids))
            scores[name]['Recall'].append(metrics.recall(predicted_chunk_ids, target_chunk_ids, k=TOPK_THRESHOLD))
            scores[name]['Precision'].append(metrics.precision(predicted_chunk_ids, target_chunk_ids, k=TOPK_THRESHOLD))
                        
            f1_score = metrics.f1_score(predicted_chunk_ids, target_chunk_ids, k=TOPK_THRESHOLD)
            scores[name]['F1'].append(0 if np.isnan(f1_score) else f1_score)

            cache_ids[name].append(predicted_chunk_ids)
            cache_texts[name].append(predicted_chunks)

            if i % show_step == 0:
                process.set_postfix({name: np.median(scores) for name, scores in scores[name].items()})

        scores[name] = {name: np.median(scores) for name, scores in scores[name].items()}
        process.set_postfix(scores[name])

    return scores, cache_ids, cache_texts