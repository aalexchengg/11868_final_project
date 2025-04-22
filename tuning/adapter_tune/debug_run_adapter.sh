#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../..

# config options are either {lora, prop, lora_prop}, get from command line
CONFIG_OPTION=$1

# if config option is not an allowed value, exit
if [ "$CONFIG_OPTION" != "lora" ] && [ "$CONFIG_OPTION" != "prop" ] && [ "$CONFIG_OPTION" != "lora_prop" ]; then
    echo "Invalid config option: $CONFIG_OPTION"
    echo "Allowed options are: lora, prop, lora_prop"
    exit 1
fi

# download base model
tune download Qwen/Qwen2.5-0.5B --output-dir /tmp/Qwen2_5-0_5B-Base

tune run tuning/custom_recipes/fim/fim_adapter_recipe.py --config debug_${CONFIG_OPTION}_config.yaml
