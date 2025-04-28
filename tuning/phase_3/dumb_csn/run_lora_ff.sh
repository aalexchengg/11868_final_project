#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../../..

CONFIG_OPTION=$1

# download base model
# tune download jtromero/qwen2-0.5b-phase2-codexglue-lora-ff --output-dir /local/Qwen2_5-0_5B-Phase2
tune download lizchu413/phase1_qwen2.5_0.5b_gh --output-dir /local/Qwen2_5-0_5B-Phase1-GH


tune run tuning/custom_recipes/fim/fim_adapter_recipe.py --config ${CONFIG_OPTION}
