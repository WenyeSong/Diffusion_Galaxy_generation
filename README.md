# Galaxy Morphology Evolution via Redshift-Conditioned Diffusion Models:
# Generation and Physical Evaluation

This repository is a reproduction and extension of the paper:

**"Learning the Evolution of Physical Structure of Galaxies via Diffusion Models"**
by Andrew Lizarraga, Eric Hanchen Jiang, Jacob Nowack, Yun Qi Li, Ying Nian Wu, Bernie Boscoe, and Tuan Do
arXiv:2411.18440 — https://arxiv.org/abs/2411.18440

The original code is available at: https://github.com/astrodatalab/lizarraga_2024/tree/main

---

## Repository Structure

```
.
├── Training/                   # DDPM model training
│   ├── Training_wy.py          # Main training script (run on HPC)
│   ├── Training_wy.ipynb       # Interactive training notebook
│   ├── dataset_analysis.ipynb  # Dataset exploration: colour-redshift, SNR distribution
│   ├── figures/                # Output plots from dataset analysis
│   └── scripts/                # SLURM submission scripts (submit_s01/05/10.sh, preview)
│
├── Generate/                   # Image generation from trained DDPM
│   ├── Generate_wy.ipynb       # Main generation + visualisation notebook
│   ├── generate_batch_wy.py    # Batch generation script (HPC array jobs)
│   ├── generate_sigma.py       # Generation with specific noise level σ
│   ├── merge_eval.py           # Merges distributed evaluation outputs
│   ├── preview_sigma.py        # Quick preview of generated images
│   └── scripts/                # SLURM submission scripts (gen, eval, merge, chain)
│
├── Evaluate/                   # Morphological metric extraction via SEP
│   ├── evaluate_wy.py          # Main SEP evaluation script (HPC)
│   ├── compute_sersic_wy.py    # Proper 1D Sersic index fitting via isophote profile
│   └── submit_evaluate.sh      # SLURM submission script
│
├── Metrics/                    # Quantitative analysis and figure reproduction
│   ├── analyse_wy.py           # Reproduces paper Figures 3, 4, Table 1
│   ├── analyse_paper_real.py   # Same analysis using original paper's Real baseline
│   ├── compute_fid_wy.py       # FID computation (InceptionV3, follows Heusel 2017)
│   ├── metrics_comparison_wy.ipynb  # Main comparison: our results vs paper
│   ├── compare_sep_vs_hdf5.ipynb    # Validates SEP measurements against HDF5 catalog
│   ├── paper_comparison/            # Notebooks comparing against original paper results
│   │   ├── AndrewMetrics.ipynb      #   Reproduces paper's generated image metrics
│   │   ├── AndrewResults.ipynb      #   Reads AndrewMetrics.csv and shows distributions
│   │   ├── GeneratedResults.ipynb   #   Analysis of our generated image metrics
│   │   └── TestingMetrics.ipynb     #   Analysis of our real test image metrics
│   ├── figures/                # Output plots
│   ├── results/                # CSV outputs and logs
│   ├── evaluation_result_*/    # Per-run evaluation result archives
│   ├── fid/                    # FID feature cache and result files
│   └── scripts/                # submit_fid.sh
│
├── CNN_Redshift/               # CNN-based photometric redshift predictor
│   ├── training/
│   │   ├── train_cnn_wy_local.ipynb  # Training notebook (local/interactive)
│   │   ├── train_cnn_wy_local.py     # Same as above as .py script
│   │   └── train_cnn_wy.py           # HPC training script
│   ├── evaluate_cnn_wy.ipynb     # Evaluation: pred vs true redshift, loss analysis
│   ├── model_output/             # Checkpoints and training log CSV
│   └── submit_cnn_train.sh       # SLURM submission script
│
├── ResCNN-MorphPredictor/      # Residual CNN for galaxy morphology prediction
│   ├── model.py                # ResCNN architecture (4-output: major axis, ellipticity,
│   │                           #   isophotal area, Sersic index)
│   ├── training/
│   │   ├── train.py            #   Training script (local)
│   │   ├── train_csd3.py       #   Training script (HPC, original HDF5 labels)
│   │   └── train_sep_csd3.py   #   Training script (HPC, SEP-derived labels)
│   ├── evaluate_train_sep.py   # Evaluation on training set with SEP labels
│   ├── step1_data_exploration.ipynb   # Band visualisation, real vs generated comparison
│   ├── step2_data_cleaning.ipynb      # Label distribution, outlier removal
│   ├── step3_model.ipynb              # Model definition and architecture diagram
│   ├── step4_train.ipynb              # Training loop and loss curve
│   ├── step5_evaluate_testset.ipynb   # Evaluation on real test set
│   ├── step6_evaluate_generated.ipynb # Evaluation on DDPM generated images;
│   │                                  #   domain shift diagnosis; SNR comparison (6.11)
│   ├── figures/                # Output plots (eval panels, architecture, radial profile…)
│   ├── scripts/                # SLURM submission scripts (rescnn, sep, pipeline)
│   ├── checkpoints/            # Model weight files (best.pt + per-epoch)
│   ├── indices_*.npy           # Fixed train/val/test split indices
│   ├── normaliser.json         # Label normalisation parameters (mean/std per target)
│   ├── image_stats.json        # Real image pixel statistics (for preprocessing)
│   └── ddpm_image_stats.json   # DDPM image pixel statistics (domain-specific norm)
│
└── Misc/                       # Shared modules used across the pipeline
    ├── modules.py              # UNet_conditional_conv, EMA, SelfAttention,
    │                           #   DoubleConv/Down/Up, CosineWarmupScheduler
    ├── data_manage.py          # HDF5ImageGenerator (PyTorch Dataset, streams from HDF5)
    ├── utils.py                # save_images, setup_logging
    └── requirements.txt        # Python package dependencies
```

---

## Pipeline Overview

1. **Train** (`Training/`) — Train the redshift-conditioned DDPM on GalaxiesML HSC images
2. **Generate** (`Generate/`) — Sample 10,000 synthetic galaxy images across z = 0.1–3.5
3. **Evaluate** (`Evaluate/`) — Extract SEP morphological metrics from real and generated images
4. **Metrics** (`Metrics/`) — Reproduce paper figures and compare distributions (FID, KS, KL)
5. **CNN_Redshift** (`CNN_Redshift/`) — Train and evaluate a dual-branch CNN redshift predictor
6. **ResCNN-MorphPredictor** (`ResCNN-MorphPredictor/`) — Train a morphology CNN; assess whether DDPM preserves realistic galaxy structure

---

## Dataset

GalaxiesML Dataset (HyperSupreme-Cam / HSC):
https://datalab.astro.ucla.edu/galaxiesml.html

5-band (g, r, i, z, y) 64×64 images with spectroscopic redshifts and morphological labels.

Dependencies: `requirements.txt`

---

## AI Usage

This project was completed independently by Wenye Song. AI tools (Claude) were used solely for brainstorming, learning background astronomy concepts and technical questions, refining report language and grammar, and debugging assistance.
