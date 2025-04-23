#!/bin/bash
#SBATCH --job-name=push
#SBATCH --output=out/push.out
#SBATCH --error=out/push.err
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --nodes=1

# Your job commands go here
date

export HF_HOME=/data/user_data/abcheng/



# Load in the correct environment
eval "$(conda shell.bash hook)"
conda activate finetune


python3 -m prune_dataset -token <hf_token>  -size 1000
