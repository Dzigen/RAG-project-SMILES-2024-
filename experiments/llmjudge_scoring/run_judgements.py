import os
import argparse
from run_judgement import PipelineArguments, run_pipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--api_key_path", type=str, required=True)
    parser.add_argument("--prompt_ending", type=str, required=True)
    parser.add_argument("--exp_nums", type=list, nargs="+", default=None)
    args = parser.parse_args()
    return args


def main():
    save_exp_dir = "/home/jovyan/work/alexander_workspace/exp/"
    save_prefix = "lmjudged_raw"
    data_prefix = "answer_labeling"

    cli_args = get_args()

    logs_dir = os.path.join(save_exp_dir, cli_args.dataset_name, "results")
    gen_args_path = os.path.join(save_exp_dir, cli_args.dataset_name, "configs/gen_args.json")

    exp_nums = cli_args.exp_nums or sorted(
        map(lambda x: os.path.splitext(x)[0][-7:], 
            filter(lambda x: x.startswith(data_prefix) and save_prefix not in x, os.listdir(logs_dir))
        )
    )
    for exp_num in exp_nums:
        data_path = os.path.join(logs_dir, data_prefix + "_" + exp_num + ".csv")
        save_path = os.path.join(logs_dir, data_prefix + "_" + save_prefix + "_" + exp_num + ".csv")
        args = PipelineArguments(
            data_path=data_path,
            save_path=save_path,
            api_key_path=cli_args.api_key_path,
            prompt_ending=cli_args.prompt_ending,
            gen_args_path=gen_args_path,
            nrows=2000
        )
        run_pipeline(args)


if __name__ == "__main__":
    main()