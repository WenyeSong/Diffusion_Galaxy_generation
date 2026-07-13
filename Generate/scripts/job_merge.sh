#!/bin/bash
#SBATCH -J merge_s${SIGMA}
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output=/rds/user/ws452/hpc-work/lizarraga_2024/logs/merge_s${SIGMA}_%j.log

PY=/home/ws452/.conda/envs/galaxy/bin/python
CODE=/rds/user/ws452/hpc-work/lizarraga_2024/code

echo "=== merge eval s${SIGMA} ==="
$PY $CODE/Generate/merge_eval.py ${SIGMA} 4
echo "=== Done ==="
