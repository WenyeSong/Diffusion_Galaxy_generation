#!/bin/bash
#SBATCH -J cnn_redshift
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o /rds/user/ws452/hpc-work/lizarraga_2024/logs/cnn_train_%j.out
#SBATCH -e /rds/user/ws452/hpc-work/lizarraga_2024/logs/cnn_train_%j.err

module load miniconda/3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate galaxy

cd /rds/user/ws452/hpc-work/lizarraga_2024/code/RedshiftCNN

/home/ws452/.conda/envs/galaxy/bin/python train_cnn_wy.py
