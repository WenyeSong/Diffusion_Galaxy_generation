#!/bin/bash
# ── Step 1 of SEP-relabelling pipeline ────────────────────────────────────────
# Computes SEP labels for training images using a 20-way array job,
# then a 4-way array job for validation images.
#
# Usage:
#   sbatch --export=SPLIT=train,N_CHUNKS=20 submit_train_sep.sh
#   sbatch --export=SPLIT=val,N_CHUNKS=4   submit_train_sep.sh
#
# After both finish, merge and retrain with train_sep_csd3.py.

#SBATCH -J sep_labels
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-19         # adjust upper bound to match N_CHUNKS-1
#SBATCH -o /rds/user/ws452/hpc-work/lizarraga_2024/logs/sep_%x_%A_%a.out
#SBATCH -e /rds/user/ws452/hpc-work/lizarraga_2024/logs/sep_%x_%A_%a.err

SPLIT=${SPLIT:-train}
N_CHUNKS=${N_CHUNKS:-20}

echo "SPLIT=$SPLIT  CHUNK=$SLURM_ARRAY_TASK_ID / $N_CHUNKS"

/home/ws452/.conda/envs/galaxy/bin/python \
    /rds/user/ws452/hpc-work/lizarraga_2024/code/ResCNN-MorphPredictor/evaluate_train_sep.py \
    --split    "$SPLIT" \
    --chunk    "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS"
