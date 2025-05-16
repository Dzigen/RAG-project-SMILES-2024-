#!/bin/bash

base_dir="/home/jovyan/work/RAG-project-SMILES-2024-/experiments/reader_context_noising_with_scores (exp #5)/with_relevant_contexts_without_score (exp #5.4)"

available_configs=`ls "$base_dir/configs"`

echo $available_configs

for current_config in $available_configs
do
    config_path="$base_dir/configs/$current_config"
    echo "Current config: $config_path"
    /home/jovyan/work/alexander_workspace/conda/rag_alex/bin/python "$base_dir/run_config.py" "$config_path"
done
