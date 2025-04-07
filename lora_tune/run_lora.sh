#!/bin/bash

tune download Qwen/Qwen2.5-0.5B --output-dir /tmp/Qwen2_5-0_5B-Base

tune run lora_tune/recipe.py --config lora_tune/config.yaml --base-model-path /tmp/Qwen2_5-0_5B-Base
