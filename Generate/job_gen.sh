#!/bin/bash
#SBATCH -J gen_s${SIGMA}
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/rds/user/ws452/hpc-work/lizarraga_2024/logs/gen_s${SIGMA}_%j.log

PY=/home/ws452/.conda/envs/galaxy/bin/python
CODE=/rds/user/ws452/hpc-work/lizarraga_2024/code

echo "=== Generate sigma=${SIGMA} ==="
$PY $CODE/Generate/generate_sigma.py ${SIGMA}
echo "=== Done ==="
