import os
import json
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from copy import deepcopy
from time import sleep
from dataclasses import dataclass

from judges import MistralJudge
from utils import make_prompt

def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        help="Путь до данных в CSV формате",
        required=True
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Если параметр задан, то будут обработаны только первые nrows строк в файле."
    )

    parser.add_argument(
        "--human_assessment",
        action="store_true",
        help="Необходимо передать этот аргумент, если в файле присутствует столбец с человеческой разметкой."
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API ключ для обращения к модели-судье."
    )

    parser.add_argument(
        "--api_key_path",
        type=str,
        default=None,
        help="Путь до файла, где сохранен API ключ для обращения к модели-судье."
    )

    parser.add_argument(
        "--gen_args_path",
        type=str,
        default="/home/jovyan/work/alexander_workspace/exp/mtsquad/configs/gen_args.json",
        help="В gen_args хранятся system, user, assistant промпты, а также некоторые параметры, передающиеся в API."
    )

    parser.add_argument(
        "--save_path",
        type=str,
        help="Путь для сохранения оценок модели в csv формате.",
        required=True
    )

    parser.add_argument(
        "--prompt_ending",
        type=str,
        required=True
    )
    
    args = parser.parse_args()

    if args.api_key is not None and args.api_key_path is not None:
        raise ValueError("Arguments --api_key and --api_key_path are provided simultaneously! You can pass only one of them!")

    if args.api_key is None and args.api_key_path is None:
        raise ValueError("The API key for LLM-judge has not provided. Please, specify --api_key or --api_key_path.")

    return args

def main():

    args = get_args()
    data = pd.read_csv(args.data_path, nrows=args.nrows)
    if args.human_assessment:
        data.drop(columns="Human assessment", inplace=True)
        
    api_key = args.api_key
    if args.api_key_path is not None:
        api_key = open(args.api_key_path, "r").read().strip()
    judge_args = dict(
        client_args = dict(
            api_key = api_key
        )
    )
    
    gen_args = json.load(open(args.gen_args_path, "r"))

    judge = MistralJudge(args=judge_args)

    limit = 10

    if not os.path.exists(args.save_path):
        pd.DataFrame({"model_assessment": []}).to_csv(args.save_path, index=False)
            
    for ind, sample in tqdm(data.iterrows()):
        user_prompt = make_prompt(gen_args["user_prompt"], ending=args.prompt_ending, *sample)
        new_gen_args = deepcopy(gen_args)

        new_gen_args["user_prompt"] = user_prompt
        response = None
        num_retries = 0
        while response is None and num_retries < limit:
            try:
                response = judge(new_gen_args)
            except Exception as e:
                print(e)
                sleep(10)
            num_retries += 1
        pd.DataFrame({"model_assessment": [response]}).to_csv(args.save_path, mode="a", header=False, index=False)


@dataclass
class PipelineArguments:
    data_path: str = None
    nrows: int = None
    human_assessment: bool = False
    api_key: str = None
    api_key_path: str = None
    gen_args_path: str = "/home/jovyan/work/alexander_workspace/exp/mtsquad/configs/gen_args.json"
    save_path: str = None
    prompt_ending: str = None


def run_pipeline(args):
    data = pd.read_csv(args.data_path, nrows=args.nrows)
    if args.human_assessment:
        data.drop(columns="Human assessment", inplace=True)
        
    api_key = args.api_key
    if args.api_key_path is not None:
        api_key = open(args.api_key_path, "r").read().strip()
    judge_args = dict(
        client_args = dict(
            api_key = api_key
        )
    )
    
    gen_args = json.load(open(args.gen_args_path, "r"))

    judge = MistralJudge(args=judge_args)

    limit = 10

    if not os.path.exists(args.save_path):
        pd.DataFrame({"model_assessment": []}).to_csv(args.save_path, index=False)
    num_processed_samples = len(pd.read_csv(args.save_path))
            
    for ind, sample in tqdm(data.iloc[num_processed_samples:].iterrows()):
        user_prompt = make_prompt(gen_args["user_prompt"], ending=args.prompt_ending, *sample)
        new_gen_args = deepcopy(gen_args)
        new_gen_args["user_prompt"] = user_prompt
        response = None
        num_retries = 0
        while response is None and num_retries < limit:
            try:
                response = judge(new_gen_args)
            except Exception as e:
                print(e)
                sleep(10)
            num_retries += 1
        if num_retries == limit:
            return
        pd.DataFrame({"model_assessment": [response]}).to_csv(args.save_path, mode="a", header=False, index=False)
    
    

if __name__ == "__main__":
    main()

    
    



