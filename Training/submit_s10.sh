#!/bin/bash
#SBATCH -J ddpm_s10
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH -o /rds/user/ws452/hpc-work/lizarraga_2024/logs/train_s10_%j.out
#SBATCH -e /rds/user/ws452/hpc-work/lizarraga_2024/logs/train_s10_%j.err

cd /rds/user/ws452/hpc-work/lizarraga_2024/code

/home/ws452/.conda/envs/galaxy/bin/python Training_wy.py 1.0
