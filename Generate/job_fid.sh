#!/bin/bash
#SBATCH -J fid_s${SIGMA}
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/ws452/hpc-work/lizarraga_2024/logs/fid_s${SIGMA}_%j.log

PY=/home/ws452/.conda/envs/galaxy/bin/python
CODE=/rds/user/ws452/hpc-work/lizarraga_2024/code

echo "=== FID sigma=${SIGMA} ==="
$PY $CODE/Metrics/compute_fid_wy.py --sigma ${SIGMA}
echo "=== Done ==="
