import os
import json

#
def save_rag_trial_log(log_dir: str, reader_scores: dict, retriever_scores: dict, save_hyperp_file: str, 
                       save_readcache_file: str, save_retrcache_file, reader_cache: dict, 
                       retriever_cache: dict, benchmarks_info: dict, benchmarks_maxsize: int, 
                       reader_params: dict, retriever_params: dict):

    if os.path.exists(log_dir):
        print("Директория существует!")
        raise ValueError
    else:
        os.mkdir(log_dir)

    hyperp_data = {'info': benchmarks_info, 
                'benchmark_sizes': benchmarks_maxsize,
                'reader_params': reader_params,
                'reader_scores': reader_scores,
                'retriever_params': retriever_params,
                'retriever_scores': retriever_scores
                }

    with open(save_hyperp_file, 'w', encoding='utf-8') as fd:
        fd.write(json.dumps(hyperp_data, indent=1))
    with open(save_readcache_file, 'w', encoding='utf-8') as fd:
        fd.write(json.dumps(reader_cache, indent=1))
    with open(save_retrcache_file, 'w', encoding='utf-8') as fd:
        fd.write(json.dumps(retriever_cache, indent=1))    

#
def save_reader_trial_log(log_dir: str, reader_scores: dict, save_hyperp_file: str, 
                          save_readcache_file: str, reader_cache: dict, benchmarks_info: dict, 
                          benchmarks_maxsize: int, reader_params: dict):
    
    if os.path.exists(log_dir):
        print("Директория существует!")
        raise ValueError
    else:
        os.mkdir(log_dir)

    hyperp_data = {'info': benchmarks_info, 
                'benchmark_sizes': benchmarks_maxsize,
                'reader_params': reader_params,
                'reader_scores': reader_scores}

    with open(save_hyperp_file, 'w', encoding='utf-8') as fd:
        fd.write(json.dumps(hyperp_data, indent=1))
    with open(save_readcache_file, 'w', encoding='utf-8') as fd:
        fd.write(json.dumps(reader_cache, indent=1))

#
def save_retriever_trial_log(log_dir:str, retriever_scores: dict, save_hyperp_file: str, 
                             save_retrcache_file: str, retriever_cache: dict, benchmarks_info: dict, 
                             benchamrks_maxsize: int, retrievers_params: dict):
    if os.path.exists(log_dir):
        print("Директория существует!")
        raise ValueError
    else:
        os.mkdir(log_dir)

    hyperp_data = {'info': benchmarks_info,
                'retriever_params': retrievers_params, 
                'benchmarks_maxsize': benchamrks_maxsize,
                'retriever_scores': retriever_scores}

    with open(save_retrcache_file, 'w', encoding='utf-8') as fd:
        fd.write(json.dumps(retriever_cache, indent=1))    
    with open(save_hyperp_file, 'w', encoding='utf-8') as fd:
        fd.write(json.dumps(hyperp_data, indent=1))