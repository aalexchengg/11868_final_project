#!/bin/bash
#SBATCH --job-name=filter_test
#SBATCH --output=filter.out
#SBATCH --error=filter.err
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=32G


# Your job commands go here


# Load in the correct environment
eval "$(conda shell.bash hook)"
conda activate finetune

python3 -m filter_exp