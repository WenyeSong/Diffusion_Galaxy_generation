#!/bin/bash
#SBATCH -J fid_compute
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /rds/user/ws452/hpc-work/lizarraga_2024/logs/fid_%j.out
#SBATCH -e /rds/user/ws452/hpc-work/lizarraga_2024/logs/fid_%j.err

cd /rds/user/ws452/hpc-work/lizarraga_2024/code

/home/ws452/.conda/envs/galaxy/bin/python compute_fid_wy.py
