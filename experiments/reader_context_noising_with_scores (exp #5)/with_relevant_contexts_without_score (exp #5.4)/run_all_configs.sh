#!/bin/bash

available_configs=`ls "./configs"`

echo $available_configs

for current_config in $available_configs
do
    config_path="./configs/$current_config"
    echo "Current config: $config_path"
    python3 run_config.py $config_path

done
