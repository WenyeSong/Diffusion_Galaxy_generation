#!/usr/bin/env python3
"""
Merge partial evaluation CSVs into one final CSV.
Usage: python merge_eval.py <SIGMA_TAG> <N_CHUNKS>
"""
import sys, os, glob
import pandas as pd

SIGMA_TAG = sys.argv[1]
N_CHUNKS  = int(sys.argv[2])
BASE      = '/rds/user/ws452/hpc-work/lizarraga_2024'
suffix    = f'_s{SIGMA_TAG}' if SIGMA_TAG else ''
OUT_DIR   = os.path.join(BASE, f'eval_output{suffix}')

parts = []
for i in range(N_CHUNKS):
    path = os.path.join(OUT_DIR, f'generated_metrics_chunk{i}.csv')
    if os.path.exists(path):
        parts.append(pd.read_csv(path))
        print(f"  loaded chunk {i}: {len(parts[-1])} rows")
    else:
        print(f"  WARNING: chunk {i} missing: {path}")

merged = pd.concat(parts, ignore_index=True)
out_path = os.path.join(OUT_DIR, f'generated_images_metrics{suffix}.csv')
merged.to_csv(out_path, index=False)
print(f"Merged {len(merged)} rows → {out_path}")
