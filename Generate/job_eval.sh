#!/bin/bash
#SBATCH -J eval_s${SIGMA}
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --time=04:00:00
#SBATCH --output=/rds/user/ws452/hpc-work/lizarraga_2024/logs/eval_s${SIGMA}_%j.log

PY=/home/ws452/.conda/envs/galaxy/bin/python
CODE=/rds/user/ws452/hpc-work/lizarraga_2024/code

echo "=== Evaluate sigma=${SIGMA} ==="
$PY $CODE/Evaluate/evaluate_wy.py --sigma ${SIGMA}
echo "=== Done ==="
