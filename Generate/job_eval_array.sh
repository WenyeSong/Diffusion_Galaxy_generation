#!/bin/bash
#SBATCH -J eval_s${SIGMA}
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=/rds/user/ws452/hpc-work/lizarraga_2024/logs/eval_s${SIGMA}_%a_%j.log

PY=/home/ws452/.conda/envs/galaxy/bin/python
CODE=/rds/user/ws452/hpc-work/lizarraga_2024/code

echo "=== eval s${SIGMA} chunk=${SLURM_ARRAY_TASK_ID}/4 ==="
$PY $CODE/Generate/evaluate_wy.py \
    --sigma ${SIGMA} \
    --chunk ${SLURM_ARRAY_TASK_ID} \
    --n_chunks 4 \
    --skip_test
echo "=== Done ==="
