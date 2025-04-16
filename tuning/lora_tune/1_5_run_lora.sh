#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../..

tune download Qwen/Qwen2.5-1.5B --output-dir /tmp/Qwen2_5-1_5B-Base

tune run tuning/custom_recipes/fim/fim_lora_recipe.py --config 1_5b_config.yaml
