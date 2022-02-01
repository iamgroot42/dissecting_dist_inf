#!/bin/bash

## Run command as:
## sbatch --export=ALL,SPLIT='victim',RATIO=0.4 --job-name=victim_0.4 bulk_train.sbatch

# Number of tasks (will be 1 unless you split job and want parallel execution)
#SBATCH --ntasks=1

# Allocation group
#SBATCH -A uvasrg

# Total memory requested on system
# Maximum 4 nodes per job allowed
# Maximum 32GB/core allowed
#SBATCH --mem=32G

# Partition of machine
#SBATCH -p gpu

# Which GPU (and how many) to request
# Prefer V100 over 2080
#SBATCH --gres=gpu:rtx2080:1

# Request specific number of CPUs for the task
# Maximum 10 cores per job allowed
#SBATCH --cpus-per-task=10

# Time limit (go for max allowed: 3 days for GPU)
#SBATCH --time=3-00:00:00

CUDA_VISIBLE_DEVICES=0 knocky bash bulk_train.sh $SPLIT $RATIO