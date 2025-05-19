import os
import json
import pandas as pd


project_dir = "path to repository"
dataset_names = ["mlqa/de", "mlqa/hi", "mlqa/vi"]
exp_series_name = "reader_context_noising_with_scores (exp #5)"
exp_names = [
    # "relevant_context_have_higher_score (exp #5.1)",
    "relevant_context_have_lower_score (exp #5.2)",
    # "with_relevant_contexts_without_score (exp #5.4)"
]
logs_dir_name = "logs"
exp_nums = ["v3.1.1", "v3.1.2"]
save_exp_dir = "exp/"
save_prefix = "answer_labeling"

for dataset_name in dataset_names:
    os.makedirs(os.path.join(save_exp_dir, dataset_name), exist_ok=True)
    for exp_name in exp_names:
        qa_dataset_path = os.path.join(project_dir, "data", dataset_name, "qa_dataset.csv")
        logs_dir_path = os.path.join(project_dir, "experiments", exp_series_name, exp_name, dataset_name, logs_dir_name)
        exp_nums = exp_nums or sorted(filter(lambda x: x.startswith("v"), os.listdir(logs_dir_path)))
        save_dir = os.path.join(save_exp_dir, dataset_name, "results")
        os.makedirs(save_dir, exist_ok=True)

        qa_dataset = pd.read_csv(qa_dataset_path)
        csv_data = pd.DataFrame()
        csv_data["Question"] = qa_dataset["question"]
        csv_data["Ground_truth_ans"] = qa_dataset["answer"]

        exp_series_num = exp_name[-2]
        
        for exp_num in exp_nums:
            exp_data = pd.read_json(os.path.join(logs_dir_path, exp_num, "generation_info.json"))
            csv_data["Gen_answer"] = exp_data["gen_answer"]
            new_exp_num = exp_num.replace('.', '_')[1:]
            save_path = os.path.join(save_dir, save_prefix + "_" + exp_series_num + "_" + new_exp_num + ".csv")
            csv_data.to_csv(save_path, index=False)