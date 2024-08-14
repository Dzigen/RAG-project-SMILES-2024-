from .utils import ThresholdRetrieverConfig, RerankRetrieverConfig
from src.Retriever import ThresholdRetriever
from src.Scorer import LLM_UncertaintyScorer
from src.Reader import LLM_Model 

from typing import List
from tqdm import tqdm

class RerankRetriever:
    def __init__(self, config: RerankRetrieverConfig, reader: LLM_Model):
        self.config = config
        self.stage1_retriever = ThresholdRetriever(self.config.stage1_retriever_config)
        self.scorer = LLM_UncertaintyScorer(self.config.scorer_config, reader)

    def rerank(self, query: str, docs: List[List[object]]):
        predicted_scores = self.scorer.predict(query, list(map(lambda doc: doc[1], docs)))
        relevant_docs = [doc for i, doc in enumerate(docs) if predicted_scores[i] == self.scorer.config.cert_label]
        return relevant_docs 
    
    def invoke(self, query: str):
        stage1_relevant_docs = self.stage1_retriever.invoke(query)
        stage2_relevant_docs = self.rerank(query, stage1_relevant_docs)

        print(len(stage1_relevant_docs), len(stage2_relevant_docs))
        
        return stage2_relevant_docs
        
        