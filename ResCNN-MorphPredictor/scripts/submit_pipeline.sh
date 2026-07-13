#!/bin/bash
# 一键提交完整流程：SEP标签计算 → 重新训练
# 用 SLURM dependency 自动保证顺序，无需手动等待
#
# 用法：bash submit_pipeline.sh

set -e

cd /rds/user/ws452/hpc-work/lizarraga_2024/code/ResCNN-MorphPredictor

# ── Step 1a：训练集 SEP（20个chunk并行）────────────────────────────────────────
JOB_TRAIN=$(sbatch --array=0-19 \
                   --export=SPLIT=train,N_CHUNKS=20 \
                   --parsable \
                   submit_train_sep.sh)
echo "训练集 SEP job 已提交：$JOB_TRAIN"

# ── Step 1b：验证集 SEP（4个chunk并行）────────────────────────────────────────
JOB_VAL=$(sbatch --array=0-3 \
                 --export=SPLIT=val,N_CHUNKS=4 \
                 --parsable \
                 submit_train_sep.sh)
echo "验证集 SEP job 已提交：$JOB_VAL"

# ── Step 2：重新训练（等两个SEP job都成功完成后自动触发）───────────────────────
JOB_TRAIN_CNN=$(sbatch --dependency=afterok:${JOB_TRAIN}:${JOB_VAL} \
                       --parsable \
                       submit_rescnn_sep.sh)
echo "训练 job 已提交：$JOB_TRAIN_CNN（等待 $JOB_TRAIN 和 $JOB_VAL 完成后自动开始）"

echo ""
echo "查看进度：squeue -u ws452"
echo "查看日志：tail -f /rds/user/ws452/hpc-work/lizarraga_2024/logs/rescnn_sep_${JOB_TRAIN_CNN}.out"
