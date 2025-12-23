# EEG-Neurodiffusion

**Topology-Preserving Diffusion Models for EEG Topomaps and Cognitive Load Classification**

EEG-Neurodiffusion is a diffusion-based generative modeling framework for EEG topographic representations (EEG topomaps) and EEG-derived images, with applications in data augmentation, cross-subject generalization, and cognitive load classification.

This repository contains the **complete research codebase** used in the paper:

> **EEG-Neurodiffusion: Topology-Preserving Diffusion Models with Vision Transformers for Cognitive Load Classification**

---

## Overview

Electroencephalography (EEG) signals are characterized by low signal-to-noise ratio, inter-subject variability, and strong spatial dependencies. While GAN-based augmentation has been explored for EEG data, such approaches often suffer from mode collapse, unstable training, and limited diversity.

**EEG-Neurodiffusion** explores *denoising diffusion probabilistic models (DDPMs)* as a stable and principled alternative for high-fidelity EEG synthesis, with a particular focus on:

- EEG scalp topographic maps (topomaps)
- Topology-preserving generation
- Class-conditional diffusion
- Controlled synthetic data injection
- Robust evaluation under frozen checkpoints

The framework is explicitly designed to study **when diffusion helps, when it hurts, and how quality control mechanisms mitigate negative transfer**.

---

## Key Contributions

- Topology-preserving EEG diffusion using conditional DDPMs  
- U-Net–based diffusion backbone  
- Multi-band EEG topomaps encoded as image channels  
- Synthetic data quality control strategies:
  - Raw diffusion augmentation  
  - Selective (class-aware) diffusion injection  
  - Classifier-guided filtering (J2)  
- Frozen-checkpoint ablation protocol for fair comparison  
- Vision Transformer (ViT) downstream classifier  
- Journal-grade statistical, degradation, and interpretability analysis  

---

## Repository Structure

EEG-Neurodiffusion/
│
├── src/ # Core diffusion & evaluation code
│ ├── diffusion/ # DDPM + U-Net implementation
│ ├── vit/ # Vision Transformer classifier
│ ├── metrics/ # MMD, PSD similarity, SSIM
│ └── eval/ # Evaluation utilities
│
├── paper/
│ ├── tables/ # LaTeX-ready tables
│ ├── journal_outputs/ # CSV outputs from analysis scripts
│ ├── journal_figs/ # Confusion matrices, ROC, attention maps
│ └── main.tex # Journal manuscript
│
├── checkpoints/ # Saved diffusion and ViT checkpoints
├── datasets/ # Preprocessed EEG topomaps (not included)
├── scripts/ # Training and generation scripts
└── README.md


---

## Reproducibility & Paper Mapping

This repository is structured to **exactly reproduce all tables and figures reported in the paper without retraining**.

| Paper Component | Script |
|-----------------|--------|
| Ablation Table | `journal_stats_from_classification_report.py` |
| Degradation Table | `journal_degradation_analysis.py` |
| Effect Size Table | `journal_effect_size_tables.py` |
| Bootstrap Confidence Intervals | `journal_stats_from_classification_report.py` |
| Confusion Matrices | `eval_confusion_matrices.py` |
| ROC / AUC Curves | `eval_roc_curves.py` |
| ViT Attention Rollout | `eval_attention_rollout.py` |

All evaluations use **frozen checkpoints** and a **fixed real test split**.

---

## Evaluation Protocol (Frozen-Checkpoint)

To isolate the effect of diffusion-based augmentation strategies, we adopt a **frozen-checkpoint evaluation protocol**:

- Each augmentation regime is trained once
- All comparisons use saved model checkpoints
- No retraining is required to reproduce results
- Subject-independent data splits are enforced
- Synthetic samples are generated **only from training data**
- No validation or test data leakage occurs

This protocol ensures **fair, stable, and reproducible comparisons**.

---

## Key Findings (Summary)

- Naive diffusion augmentation is **not automatically beneficial** and can significantly degrade EEG classification performance  
- Explicit quality control is essential when using diffusion-generated EEG samples  
- Selective diffusion provides the most reliable trade-off, preserving most real-only performance  
- Classifier-guided filtering reduces harmful outliers but may underutilize informative diversity  
- Diffusion augmentation should be framed as **damage control rather than guaranteed improvement**

---

## Notes

⚠ **Important:**  
Diffusion-based augmentation for EEG should always be paired with explicit quality control. Uncontrolled synthetic injection may distort class-conditional manifolds and reduce generalization.

---

## Requirements

- Python ≥ 3.9  
- PyTorch ≥ 1.13  
- NumPy, SciPy, scikit-learn  
- matplotlib  
- timm (Vision Transformer models)

Exact versions used in the paper are listed in `requirements.txt`.
