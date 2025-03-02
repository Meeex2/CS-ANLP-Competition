#!/bin/bash
#SBATCH --job-name=nlp_test_mbert  # Job name
#SBATCH --nodes=1                                  # Use 1 node
#SBATCH --partition=gpu_test                # Use the GPU partition
#SBATCH --gres=gpu:4                             # Request 2 GPUs
#SBATCH --time=01:00:00                            # Time limit
#SBATCH --mem=32G
#SBATCH --output=nlp_test_mbert.out       # Standard output (with job ID)
#SBATCH --error=nlp_test_mbert.err        # Standard error (with job ID)
#SBATCH --mail-type=ALL                            # Send email on start, end and fail

micromamba shell init --shell=bash
eval "$(micromamba shell hook -s bash)"
micromamba activate multilang-code-vul-detection 

# Load necessary modules
module load cuda/12.2.1/gcc-11.2.0

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Start GPU monitoring in background; log full nvidia-smi output every 30 seconds to gpu_usage.log.
nvidia-smi -l 60 > gpu_usage_mbert.log 2>&1 &
MONITOR_PID=$!

# Run the training script with accelerate using 2 processes.
accelerate launch --multi_gpu --mixed_precision "fp16" --num_processes 4 scripts/continue_mbert.py > nlp_test_mbert${SLURM_JOB_ID}.log
# version for single gpu
# accelerate launch --mixed_precision "fp16" --num_processes 1 scripts/train_mbert.py > nlp_test_mbert${SLURM_JOB_ID}.log

# After training completes, stop the GPU monitor.
kill $MONITOR_PID
