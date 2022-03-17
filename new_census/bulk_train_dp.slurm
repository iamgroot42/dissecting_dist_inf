#!/bin/bash

# Run command as:
## sbatch --export=ALL,SPLIT='victim',FILTER='sex',EPSILON=0.1,RATIO=0.0 --job-name=vics1 bulk_train_dp.slurm

# Number of tasks (will be 1 unless you split job and want parallel execution)
#SBATCH --ntasks=1

# Allocation group
#SBATCH -A uvasrg

# Total memory requested on system
#SBATCH --mem=10G

# Which GPU (and how many) to request
#SBATCH --gres=gpu:v100:1

# Request specific number of CPUs for the task
#SBATCH --cpus-per-task=10

# Time limit (go for max allowed: 3 days for GPU)
#SBATCH --time=3-00:00:00

# Output file paths
#SBATCH --output=log/training/%x-%j.out
#SBATCH --error=log/training/%x-%j.err

export CUDA_VISIBLE_DEVICES=0;
knocky python train_dp_models.py --split $SPLIT --filter $FILTER --num 1000 --epsilon $EPSILON --ratio $RATIO
