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


@dataclass
class PipelineArguments:
    data_path: str = None
    nrows: int = None
    human_assessment: bool = False
    api_key: str = None
    api_key_path: str = None
    gen_args_path: str = "exp/gen_args.json"
    save_path: str = None
    prompt_ending: str = None


def run_pipeline(args: PipelineArguments):
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