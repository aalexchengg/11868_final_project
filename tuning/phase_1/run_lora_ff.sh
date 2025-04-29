#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../..
export HF_DATASETS_CACHE="/local/cache/huggingface"

CONFIG_OPTION=$1

# download base model
tune download Qwen/Qwen2.5-0.5B --output-dir /local/Qwen2_5-0_5B-Base

tune run tuning/custom_recipes/fim/fim_adapter_recipe.py --config ${CONFIG_OPTION}
