import pandas as pd
import ast

from src.Retriever.utils import ThresholdRetrieverConfig
from src.Reader.utils import LLM_Hardw_Conf, LLM_Hyper_Conf

def load_benchmarks_df(benchmarks_path: dict, benchmarks_maxsize: int) -> dict:  
    benchmarks_df = {}
    for name, _ in benchmarks_path.items():
        benchmarks_df[name] = pd.read_csv(benchmarks_path[name]['table'], sep=';').iloc[:benchmarks_maxsize,:]
        benchmarks_df[name]['chunk_ids'] = benchmarks_df[name]['chunk_ids'].map(lambda v: ast.literal_eval(v)) 
        benchmarks_df[name]['contexts'] = benchmarks_df[name]['contexts'].map(lambda v: ast.literal_eval(v)) 

    return benchmarks_df

def prepare_retriever_configs(base_dir: str, benchmarks_info: dict, retriever_params: dict) -> tuple: 
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
    hardw_c = LLM_Hardw_Conf(model_path=reader_params['llm_path'], system_prompt=reader_params['system_prompt'], 
                             assistant_prompt=reader_params['assistant_prompt'])

    hyperp = reader_params.copy()
    del hyperp['system_prompt']
    del hyperp['assistant_prompt']
    del hyperp['llm_path']
    hyperp_c = LLM_Hyper_Conf(**hyperp)

    return hardw_c, hyperp_c