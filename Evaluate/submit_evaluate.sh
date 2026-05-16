#!/bin/bash
#SBATCH -J galaxy_evaluate
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o /rds/user/ws452/hpc-work/lizarraga_2024/logs/evaluate_%j.out
#SBATCH -e /rds/user/ws452/hpc-work/lizarraga_2024/logs/evaluate_%j.err

cd /rds/user/ws452/hpc-work/lizarraga_2024/code

/home/ws452/.conda/envs/galaxy/bin/python evaluate_wy.py
