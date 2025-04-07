#!/bin/bash

tune download Qwen/Qwen2.5-0.5B --output-dir /tmp/Qwen2_5-0_5B-Base

tune run recipe.py --config config.yaml
