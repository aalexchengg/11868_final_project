#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../..

# config options are either {csn, cxg, gh}, get from command line
CONFIG_OPTION=$1

# if config option is not an allowed value, exit
if [ "$CONFIG_OPTION" != "csn" ] && [ "$CONFIG_OPTION" != "cxg" ] && [ "$CONFIG_OPTION" != "gh" ]; then
    echo "Invalid config option: $CONFIG_OPTION"
    echo "Allowed options are: csn, cxg, gh"
    exit 1
fi

tune download Qwen/Qwen2.5-0.5B --output-dir /tmp/Qwen2_5-0_5B-Base
#tune download Qwen/Qwen2.5-1.5B --output-dir /tmp/Qwen2_5-1_5B-Base

tune run tuning/custom_recipes/fim/fim_vanilla_recipe.py --config config_0_5_${CONFIG_OPTION}.yaml > phase1_${CONFIG_OPTION}.txt
#tune run recipe.py --config config_1_5.yaml
