#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../..
export HF_DATASETS_CACHE="/local/cache/huggingface"

CONFIG_OPTION=$1

# download base model
tune download jtromero/qwen2-0.5b-phase1-gh_plus-lora-ff --output-dir /local/Qwen2_5-0_5B-Phase1-GH-Checkpoint

tune run tuning/custom_recipes/fim/fim_adapter_recipe.py --config ${CONFIG_OPTION}
