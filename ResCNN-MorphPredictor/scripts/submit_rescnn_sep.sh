#!/bin/bash
#SBATCH -J rescnn_sep
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o /rds/user/ws452/hpc-work/lizarraga_2024/logs/rescnn_sep_%j.out
#SBATCH -e /rds/user/ws452/hpc-work/lizarraga_2024/logs/rescnn_sep_%j.err

cd /rds/user/ws452/hpc-work/lizarraga_2024/code/ResCNN-MorphPredictor

/home/ws452/.conda/envs/galaxy/bin/python training/train_sep_csd3.py
