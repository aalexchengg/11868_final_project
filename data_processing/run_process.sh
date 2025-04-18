#!/bin/bash
#SBATCH --job-name=make_github
#SBATCH --output=out/make_github.out
#SBATCH --error=out/make_github.err
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=32G

# Your job commands go here


# Load in the correct environment
eval "$(conda shell.bash hook)"
conda activate finetune


python3 -m make_github -out aalexchengg/github_test -size 10
