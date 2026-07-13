#!/bin/bash
#SBATCH -J rescnn_morph
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /rds/user/ws452/hpc-work/lizarraga_2024/logs/rescnn_%j.out
#SBATCH -e /rds/user/ws452/hpc-work/lizarraga_2024/logs/rescnn_%j.err

cd /rds/user/ws452/hpc-work/lizarraga_2024/code/ResCNN-MorphPredictor

/home/ws452/.conda/envs/galaxy/bin/python train_csd3.py
