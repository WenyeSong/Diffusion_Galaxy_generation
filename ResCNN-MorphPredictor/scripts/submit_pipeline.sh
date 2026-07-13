#!/bin/bash
# Submit the full pipeline in one command: SEP label computation -> retrain
# SLURM dependencies ensure correct ordering automatically, no manual waiting needed
#
# Usage: bash submit_pipeline.sh

set -e

cd /rds/user/ws452/hpc-work/lizarraga_2024/code/ResCNN-MorphPredictor

# ── Step 1a: training set SEP (20 chunks in parallel) ──────────────────────────
JOB_TRAIN=$(sbatch --array=0-19 \
                   --export=SPLIT=train,N_CHUNKS=20 \
                   --parsable \
                   submit_train_sep.sh)
echo "Training set SEP job submitted: $JOB_TRAIN"

# ── Step 1b: validation set SEP (4 chunks in parallel) ─────────────────────────
JOB_VAL=$(sbatch --array=0-3 \
                 --export=SPLIT=val,N_CHUNKS=4 \
                 --parsable \
                 submit_train_sep.sh)
echo "Validation set SEP job submitted: $JOB_VAL"

# ── Step 2: retrain (triggered automatically after both SEP jobs succeed) ───────
JOB_TRAIN_CNN=$(sbatch --dependency=afterok:${JOB_TRAIN}:${JOB_VAL} \
                       --parsable \
                       submit_rescnn_sep.sh)
echo "Training job submitted: $JOB_TRAIN_CNN (will start after $JOB_TRAIN and $JOB_VAL complete)"

echo ""
echo "Check progress: squeue -u ws452"
echo "Check logs:     tail -f /rds/user/ws452/hpc-work/lizarraga_2024/logs/rescnn_sep_${JOB_TRAIN_CNN}.out"
