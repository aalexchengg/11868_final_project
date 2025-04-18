#!/bin/bash
#SBATCH --job-name=big_dataset
#SBATCH --output=out/process.out
#SBATCH --error=out/process.err
#SBATCH --partition=general
#SBATCH --gpus=1
#SBATCH --mem=32G

# Your job commands go here


# Load in the correct environment
eval "$(conda shell.bash hook)"
conda activate finetune


python3 -m codesearchnet

echo "All finished"
