# Reproduction of Galaxy Diffusion Model

This repository is a reproduction and extension of the paper:

**"Learning the Evolution of Physical Structure of Galaxies via Diffusion Models"**
by Andrew Lizarraga, Eric Hanchen Jiang, Jacob Nowack, Yun Qi Li, Ying Nian Wu, Bernie Boscoe, and Tuan Do
arXiv:2411.18440 — https://arxiv.org/abs/2411.18440

The original code is available at: https://github.com/astrodatalab/lizarraga_2024/tree/main

This reproduction is ongoing and includes additional analysis and modifications.

---

## Structure

- `Training/` — DDPM training notebooks and scripts
- `Generate/` — Image generation notebooks and batch generation scripts
- `Evaluate/` — SEP-based morphological metric extraction
- `Metrics/` — Analysis scripts reproducing Figures 3, 4, 5, 6 and Table 1
- `CNN_Redshift/` — CNN redshift predictor (adapted from Li et al. 2024)
- `Misc/` — Shared modules (model architecture, data loader, utilities)

## Dataset

Training data is the GalaxiesML Dataset (HyperSuprime-Cam):
https://datalab.astro.ucla.edu/galaxiesml.html

Requirements are listed in `Misc/requirements.txt`.
