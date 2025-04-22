#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../..

# download base model
tune download Qwen/Qwen2.5-0.5B --output-dir /local/Qwen2_5-0_5B-Base

tune run tuning/custom_recipes/fim/fim_adapter_recipe.py --config ${CONFIG_OPTION}
