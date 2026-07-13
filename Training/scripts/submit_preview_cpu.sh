#!/bin/bash
#SBATCH -J preview_sigma
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --time=00:20:00
#SBATCH --output=/rds/user/ws452/hpc-work/lizarraga_2024/logs/preview_%j.log

PY=/home/ws452/.conda/envs/galaxy/bin/python
CODE=/rds/user/ws452/hpc-work/lizarraga_2024/code

$PY $CODE/Generate/preview_sigma.py 01
$PY $CODE/Generate/preview_sigma.py 05
$PY $CODE/Generate/preview_sigma.py 10

echo "All previews done."
