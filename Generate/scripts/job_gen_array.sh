#!/bin/bash
#SBATCH -J gen_s${SIGMA}
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --array=0-9
#SBATCH --output=/rds/user/ws452/hpc-work/lizarraga_2024/logs/gen_s${SIGMA}_%a_%j.log

PY=/home/ws452/.conda/envs/galaxy/bin/python
CODE=/rds/user/ws452/hpc-work/lizarraga_2024/code

CHUNK_SIZE=1000
START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))

echo "=== gen s${SIGMA} array=${SLURM_ARRAY_TASK_ID} images ${START}-${END} ==="
$PY $CODE/Generate/generate_sigma.py ${SIGMA} --start $START --end $END
echo "=== Done ==="
