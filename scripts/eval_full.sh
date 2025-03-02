#!/bin/bash
#SBATCH --job-name=modern_bert_finetune_multi_gpu  # Job name
#SBATCH --nodes=1                                  # Use 1 node
#SBATCH --partition=gpu_test                   # Use the GPU partition
#SBATCH --gres=gpu:1                              # Request 2 GPUs
#SBATCH --time=01:00:00                            # Time limit
#SBATCH --mem=32G
#SBATCH --output=modern_bert_finetune_test1_%j.out       # Standard output (with job ID)
#SBATCH --error=modern_bert_finetune_test1_%j.err        # Standard error (with job ID)
#SBATCH --mail-type=ALL                            # Send email on start, end and fail

micromamba shell init --shell=bash
eval "$(micromamba shell hook -s bash)"
micromamba activate multilang-code-vul-detection 

# Load necessary modules
module load cuda/12.2.1/gcc-11.2.0

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Start GPU monitoring in background; log full nvidia-smi output every 30 seconds to gpu_usage.log.
nvidia-smi -l 60 > gpu_usage.log 2>&1 &
MONITOR_PID=$!

# Run the training script with accelerate using 2 processes.
accelerate launch --mixed_precision "fp16" --num_processes 1 submission.py > eval_full.log

# After training completes, stop the GPU monitor.
kill $MONITOR_PID