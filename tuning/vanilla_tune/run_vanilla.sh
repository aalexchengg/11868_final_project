#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../..

tune download Qwen/Qwen2.5-0.5B --output-dir /tmp/Qwen2_5-0_5B-Base
#tune download Qwen/Qwen2.5-1.5B --output-dir /tmp/Qwen2_5-1_5B-Base

tune run recipe.py --config config_0_5_cxg.yaml > phase1_cxg.txt
#tune run recipe.py --config config_1_5.yaml
