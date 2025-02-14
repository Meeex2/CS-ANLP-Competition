#!/bin/bash
#SBATCH --job-name=modern_bert_finetune  # Name of the job
#SBATCH --nodes=1                        # Use 1 node
#SBATCH --partition=gpu                  # Use the GPU partition
#SBATCH --gres=gpu:1                     # Request 4 GPUs
#SBATCH --time=01:00:00                  # Time limit (2 hours)
#SBATCH --output=modern_bert_finetune_%j.out  # Output file (with job ID)
#SBATCH --error=modern_bert_finetune_%j.err   # Error file (with job ID)
#SBATCH --mail-type=ALL                  # Send email on start, end and fail


micromamba shell init --shell=bash

eval "$(micromamba shell hook -s bash)"

micromamba activate multilang-code-vul-detection 

# Load necessary modules
module load cuda/12.2.1/gcc-11.2.0

# Set environment variables
export TOKENIZERS_PARALLELISM=false



# Run the training script with accelerate
accelerate launch --multi_gpu --mixed_precision "fp16" --num_processes 2 --num_machines 1 --gpu_ids "0,1" --machine_rank 0 scripts/training_script.py > modern_bert_finetune.log