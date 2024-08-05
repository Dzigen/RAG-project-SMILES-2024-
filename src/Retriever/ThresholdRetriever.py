from .utils import ThresholdRetrieverConfig
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

class ThresholdRetriever:
    def __init__(self, config: ThresholdRetrieverConfig) -> None:
        self.config = config

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.model_path,
            model_kwargs=self.config.model_kwargs,
            encode_kwargs=self.config.encode_kwargs
        )

        self.client = chromadb.PersistentClient(path=self.config.densedb_path)
        self.collection = self.client.get_or_create_collection(**self.config.densedb_kwargs)
        
    def invoke(self, query: str):
        query_embedding = self.embeddings.embed_query(query)

        docs_with_scores = self.collection.query(
            query_embeddings=[query_embedding],
            include=["documents", "metadatas", "distances"],
            n_results=self.config.params['fetch_k'])
        
        distance_metric = self.config.densedb_kwargs['metadata']["hnsw:space"]
        filtered_docs_id = list(filter(
            lambda i: docs_with_scores['distances'][0][i] < self.config.params['threshold'], 
                                                    range(len(docs_with_scores['documents'][0]))))
        
        if self.config.params['max_k'] > 0:
            filtered_docs_id = filtered_docs_id[:self.config.params['max_k']]

        return list(map(lambda i: [docs_with_scores['distances'][0][i], docs_with_scores['documents'][0][i], 
                                       docs_with_scores['metadatas'][0][i]], filtered_docs_id))  