#!/bin/bash
# Submit full pipeline chain for one or more sigma tags.
# Usage:  bash submit_chain.sh 01 05 10
#
# Per sigma, job graph:
#   gen_array (10 GPU jobs, 20min each, parallel)
#     └─ eval_array (4 CPU jobs, 1.5h each, parallel, afterok:gen_array)
#          └─ merge (1 CPU job, 10min, afterok:eval_array)
#               └─ fid (1 GPU job, 1h, afterok:merge)

CODE=/rds/user/ws452/hpc-work/lizarraga_2024/code

for SIGMA in "$@"; do
    echo "━━━ Submitting chain for sigma=${SIGMA} ━━━"

    # 1. Generate: 10 array jobs (each handles 1000 images, ~15min GPU)
    GEN_ID=$(sbatch --parsable \
        --export=SIGMA=${SIGMA} \
        $CODE/Generate/job_gen_array.sh)
    echo "  gen   array job: ${GEN_ID}  (10 tasks × 20min GPU)"

    # 2. Evaluate: 4 array jobs after ALL gen tasks finish
    EVAL_ID=$(sbatch --parsable \
        --dependency=afterok:${GEN_ID} \
        --export=SIGMA=${SIGMA} \
        $CODE/Generate/job_eval_array.sh)
    echo "  eval  array job: ${EVAL_ID}  (4 tasks × 1.5h CPU, after gen)"

    # 3. Merge CSVs after ALL eval tasks finish
    MERGE_ID=$(sbatch --parsable \
        --dependency=afterok:${EVAL_ID} \
        --export=SIGMA=${SIGMA} \
        $CODE/Generate/job_merge.sh)
    echo "  merge       job: ${MERGE_ID}  (10min CPU, after eval)"

    # 4. FID after merge
    FID_ID=$(sbatch --parsable \
        --dependency=afterok:${MERGE_ID} \
        --export=SIGMA=${SIGMA} \
        $CODE/Generate/job_fid.sh)
    echo "  fid         job: ${FID_ID}  (1h GPU, after merge)"

    echo ""
done

echo "All chains submitted. squeue -u ws452 to check."
