#!/bin/bash

#SBATCH --ntasks=1
#SBATCH -A uvasrg

# Total memory requested on system
# Maximum 4 nodes per job allowed
# Maximum 32GB/core allowed
#SBATCH --mem-per-cpu=8G

# Partition of machine
#SBATCH -p gpu

# Which GPU (and how many) to request
# Prefer V100 over 2080
#SBATCH --gres=gpu:v100:1

# Request specific number of CPUs for the task
# Maximum 10 cores per job allowed
#SBATCH --cpus-per-task=2

# Time limit (go for max allowed: 3 days for GPU)
#SBATCH --time=2-00:00:00

# Output file paths
#SBATCH --output="r%a.out"
#Need restore 1-2, 7-9, 10-12, 15-20,24,27,32-33,37-39
OFFSETS=(0 50 100 150 200 250 300 350 400 450)
RATIOS=(0.0 0.1 0.2 0.3)
# Your command goes here
echo $SLURM_ARRAY_TASK_ID
#nohup ./do_all.sh "$PARAM1" ${OFFSETS[$SLURM_ARRAY_TASK_ID]} 2>&1 > "r$SLURM_ARRAY_TASK_ID.out" &
sleep $((${OFFSETS[$(($SLURM_ARRAY_TASK_ID%10))]}/10))
python train_more.py --load_config "./cel_mouth.json" --ratios ${RATIOS[$(($SLURM_ARRAY_TASK_ID/10))]} --split adv --offset ${OFFSETS[$(($SLURM_ARRAY_TASK_ID%10))]} 
wait
echo "finished"