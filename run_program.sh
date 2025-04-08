#!/bin/bash
#SBATCH --job-name=training_args
#SBATCH --output=out/args.out
#SBATCH --error=out/args.err
#SBATCH --partition=general
#SBATCH --gpus=4

# Your job commands go here
eval "$(conda shell.bash hook)"
conda activate finetune

python test.py