import pandas as pd
import ast

from src.Retriever.utils import ThresholdRetrieverConfig, RerankRetrieverConfig
from src.Scorer import UncertaintyScorerConfig
from src.Reader.utils import LLM_Config, LLM_PromptsConfig, LLM_DataOperateConfig

def load_benchmarks_df(benchmarks_path: dict, benchmarks_maxsize: int) -> dict:  
    benchmarks_df = {}
    for name, _ in benchmarks_path.items():
        benchmarks_df[name] = pd.read_csv(benchmarks_path[name]['table'], sep=';')
        if benchmarks_maxsize > 0:
            benchmarks_df[name] = benchmarks_df[name].iloc[:benchmarks_maxsize,:]

        benchmarks_df[name]['chunk_ids'] = benchmarks_df[name]['chunk_ids'].map(lambda v: ast.literal_eval(v))

    return benchmarks_df

def prepare_rerankretriever_configs(base_dir: str, benchmarks_info: dict, retriever_params: dict):
    banchmarks_path = {}
    benchmarks_config = {}

    for name, version in benchmarks_info.items():
        banchmarks_path[name] = {
            'table': f"{base_dir}/data/{name}/tables/{version['table']}/benchmark.csv",
            'dense_db': f"{base_dir}/data/{name}/dbs/{version['db']}/densedb"
        }
    
        config = RerankRetrieverConfig(
            stage1_retriever_config=ThresholdRetrieverConfig(**retriever_params["stage1_retriever_config"]),
            scorer_config=UncertaintyScorerConfig(**retriever_params["scorer_config"]))
        config.stage1_retriever_config.densedb_path = banchmarks_path[name]['dense_db']
        config.stage1_retriever_config.densedb_kwargs['name'] = name
        benchmarks_config[name] = config

    return benchmarks_config, banchmarks_path

def prepare_thresholdretriever_configs(base_dir: str, benchmarks_info: dict, retriever_params: dict) -> tuple: 
    banchmarks_path = {}
    benchmarks_config = {}

    for name, version in benchmarks_info.items():
        banchmarks_path[name] = {
            'table': f"{base_dir}/data/{name}/tables/{version['table']}/benchmark.csv",
            'dense_db': f"{base_dir}/data/{name}/dbs/{version['db']}/densedb"
        }
    
        config = ThresholdRetrieverConfig(**retriever_params)
        config.densedb_path = banchmarks_path[name]['dense_db']
        config.densedb_kwargs['name'] = name
        benchmarks_config[name] = config

    return benchmarks_config, banchmarks_path

def prepare_reader_configs(reader_params: dict) -> tuple:
    config = LLM_Config(
        prompts=LLM_PromptsConfig(**reader_params['prompts']),
        gen=reader_params['gen'],
        data_operate=LLM_DataOperateConfig(**reader_params['data_operate'])
    )

    return config