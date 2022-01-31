#!/bin/bash

# Number of tasks (will be 1 unless you split job and want parallel execution)
#SBATCH --ntasks=1

# Allocation group
#SBATCH -A uvasrg

# Total memory requested on system
#SBATCH --mem=100G

# Partition of machine
#SBATCH -p gpu

# Which GPU (and how many) to request
# Prefer V100 over 2080
#SBATCH --gres=gpu:rtx2080:4

# Request specific number of CPUs for the task
#SBATCH --cpus-per-task=40

# Name your job
#SBATCH --job-name=celeba_victim

CUDA_VISIBLE_DEVICES=0 knocky bash bulk_train.sh adv 0.1 &
CUDA_VISIBLE_DEVICES=1 knocky bash bulk_train.sh adv 0.2 &
CUDA_VISIBLE_DEVICES=2 knocky bash bulk_train.sh adv 0.3 &
CUDA_VISIBLE_DEVICES=3 knocky bash bulk_train.sh adv 0.4 &
wait