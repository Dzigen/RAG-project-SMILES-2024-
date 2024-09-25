# Source: https://amitness.com/2020/08/information-retrieval-evaluation/

#Retrieval metrics
# - mAP
# - MRR
# - precision
# - recall
# - f1
#Reader metrics
# - BLEU presision
# - ROUGE recall
# - METEOR f1

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
from evaluate import load
import evaluate
import numpy as np
from typing import List
from tqdm import tqdm
from torchmetrics.text.bert import BERTScore
from Levenshtein import distance as levenshtain_distance

from src.Reader import LLM_Model
from src.Scorer import LLM_SimilarityScorer, SimilarityScorerConfig

#
class RetrieverMetrics:
    def __init__(self):
        pass
    
    def precision(self, predicted_cands: List[int], gold_cands: List[int], k: int) -> float:
        true_positive = np.isin(predicted_cands[:k], gold_cands).sum()
        false_positive = k - true_positive
        return round(true_positive / (true_positive + false_positive),5)

    def recall(self, predicted_cands: List[int], gold_cands: List[int], k: int) -> float:
        true_positive = np.isin(predicted_cands[:k], gold_cands).sum()
        false_negative = len(gold_cands) - true_positive
        return round(true_positive / (true_positive + false_negative),5)

    def f1_score(self, pred_cands: List[int], gold_cands: List[int], k: int) -> float:
        return (2 * self.precision(pred_cands, gold_cands, k) * self.recall(pred_cands, gold_cands, k)) / (self.precision(pred_cands, gold_cands, k) + self.recall(pred_cands, gold_cands, k))

    def AP(self, predicted_cands: List[int], gold_cands: List[int]) -> float:
        indicators = np.isin(predicted_cands, gold_cands)

        numerator = np.sum([self.precision(predicted_cands, gold_cands, k+1) 
                            for k in range(len(predicted_cands)) if indicators[k]])

        return round(numerator / len(gold_cands), 5)

    def mAP(self, predicted_cands_batch: List[List[int]], gold_cands_batch: List[List[float]]) -> float:
        return np.mean([self.AP(pred_cands, gold_cands) 
                        for pred_cands, gold_cands in zip(predicted_cands_batch, gold_cands_batch)])

    def reciprocal_rank(self, predicted_cands: List[int], gold_cands: List[int]) -> float:
        indicators = np.isin(predicted_cands, gold_cands)
        first_occur = np.where(indicators == True)[0]
        return round(0 if first_occur.size == 0 else 1 / (first_occur[0] + 1), 5)
        
    def MRR(self, predicted_cands_batch: List[List[int]], gold_cands_batch: List[List[int]]):
        return np.mean([self.reciprocal_rank(pred_cands, gold_cands) 
                        for pred_cands, gold_cands in zip(predicted_cands_batch, gold_cands_batch)])
    
# 
class ReaderMetrics:
    def __init__(self, base_dir, model_path, sim_score_config: SimilarityScorerConfig = None, reader: LLM_Model = None):
        self.rouge_obj = ROUGEScore()
        self.bleu1_obj = BLEUScore(n_gram=1)
        self.bleu2_obj = BLEUScore(n_gram=2)
        print("Loading Meteor...")
        self.meteor_obj = evaluate.load(f"{base_dir}/src/utils/metrics/meteor")
        print("Loading ExactMatch")
        self.em_obj = evaluate.load(f"{base_dir}/src/utils/metrics/exact_match")
        self.bertscore_obj = BERTScore(f"{base_dir}/models/{model_path}", return_hash=True)
        if sim_score_config is not None:
            self.llmsimscore_obj = LLM_SimilarityScorer(sim_score_config, reader)
    
    def bertscore(self, predicted: List[str], targets: List[str]):
        output = self.bertscore_obj(predicted, targets)
        output['precision'] = round(float(output['precision'].mean()), 5)
        output['recall'] = round(float(output['recall'].mean()), 5)
        output['f1'] = round(float(output['f1'].mean()), 5)
        
        return output
    
    def rougel(self, predicted: List[str], targets: List[str]):
        return [self.rouge_obj(
            predicted[i], targets[i])['rougeL_fmeasure'] 
                 for i in range(len(targets))]
        
    def bleu1(self, predicted: List[str], targets: List[str]):
        return [self.bleu1_obj(
            [predicted[i]], [[targets[i]]])
                 for i in range(len(targets))]
        
    def bleu2(self, predicted: List[str], targets: List[str]):
        return [self.bleu2_obj(
            [predicted[i]], [[targets[i]]]) 
                 for i in range(len(targets))]
        
    def meteor(self, predicted: List[str], targets: List[str]):
        return [self.meteor_obj.compute(
            predictions=[predicted[i]], references=[targets[i]])['meteor'] 
                 for i in range(len(targets))]
        
    def exact_match(self, predicted: List[str], targets: List[str]):
        return [self.em_obj.compute(
            predictions=[predicted[i]], references=[targets[i]])["exact_match"]
                for i in range(len(targets))]

    def llm_sim_score(self, texts1: List[str], texts2: List[str]):
        return self.llmsimscore_obj.predict(texts1, texts2)
    
    def levenshtain_score(self, predicted: List[str], targets: List[str]):
        return list(map(lambda pair: levenshtain_distance(pair[1], pair[0]), zip(predicted, targets)))










