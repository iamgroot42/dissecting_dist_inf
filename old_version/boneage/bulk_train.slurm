#!/bin/bash

## Run command as:
## sbatch --job-name=bone --array=0-6 bulk_train.slurm

# Number of tasks (will be 1 unless you split job and want parallel execution)
#SBATCH --ntasks=1

# Allocation group
#SBATCH -A uvasrg

# Total memory requested on system
# Maximum 4 nodes per job allowed
# Maximum 32GB/core allowed
#SBATCH --mem=64G

# Partition of machine
#SBATCH -p gpu

# Which GPU (and how many) to request
# Prefer V100 over 2080
#SBATCH --gres=gpu:rtx2080:1

# Request specific number of CPUs for the task
# Maximum 10 cores per job allowed
#SBATCH --cpus-per-task=16

# Time limit (go for max allowed: 3 days for GPU)
#SBATCH --time=3-00:00:00

# Output file paths
#SBATCH --output=log/training/%x-%j-%a.out
#SBATCH --error=log/training/%x-%j-%a.err

RATIOS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)
CUDA_VISIBLE_DEVICES=0 knocky python train_models.py --split victim --full_model --num 250 --ratio ${RATIOS[$SLURM_ARRAY_TASK_ID]}
