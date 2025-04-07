#!/bin/bash
#SBATCH --job-name=vanilla
#SBATCH --output=out/vanilla.out
#SBATCH --error=out/vanilla.err
#SBATCH --partition=general
#SBATCH --gpus=2

# Your job commands go here


# Load in the correct environment
eval "$(conda shell.bash hook)"
conda activate finetune


python3 run.py