from dataclasses import dataclass, field
from ruamel.yaml import YAML
from typing import List, Dict, Union

from src.Scorer import UncertaintyScorerConfig

# USEFUL links
# https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/
# https://stackoverflow.com/questions/77217193/langchain-how-to-use-a-custom-embedding-model-locally
# https://github.com/langchain-ai/langchain/discussions/9645
# https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/
# https://www.kaggle.com/discussions/general/509903
# https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/ensemble/
# https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html

# [threshold, fetch_ks]

@dataclass
class ThresholdRetrieverConfig:
    model_path: str = "/trinity/home/team06/workspace/mikhail_workspace/rag_project/models/intfloat/multilingual-e5-small"
    densedb_path: str = "/trinity/home/team06/workspace/mikhail_workspace/rag_project/data/mtssquad/dbs/v2/densedb"
    densedb_kwargs: Dict[str, object] = field(default_factory=lambda: {'metadata': {"hnsw:space": "ip"}, 'name': 'mtssquad'})

    encode_kwargs: Dict[str, object] = field(default_factory=lambda: {'normalize_embeddings': True, 'prompt': 'query: '})
    model_kwargs: Dict[str, object] = field(default_factory=lambda: {'device': 'cuda'})
    
    params: Dict[str, object] = field(default_factory=lambda: {'fetch_k': 50, 'threshold': 0.2, 'max_k': 10})

@dataclass
class RerankRetrieverConfig:
    stage1_retriever_config: ThresholdRetrieverConfig = field(default_factory=lambda: ThresholdRetrieverConfig())
    scorer_config: UncertaintyScorerConfig = field(default_factory=lambda: UncertaintyScorerConfig())