# 📄 Reproducibility Guide for EEG-Neurodiffusion

This document describes how to **exactly reproduce all tables, figures, and results**
reported in the paper:

> **EEG-Neurodiffusion: Topology-Preserving Diffusion Models with Vision Transformers for Cognitive Load Classification**

⚠ **No retraining is required.**  
All results are generated using **frozen model checkpoints** and a **fixed real test split**.

---

## Reproducibility Philosophy

All experiments in this work follow a **frozen-checkpoint evaluation protocol** designed
to eliminate variability and ensure exact reproducibility.

Specifically:

- Each augmentation regime is trained **once**
- Model weights are saved as checkpoints
- All reported metrics, tables, and figures are generated from these fixed checkpoints
- A single evaluation pipeline is used for all comparisons
- No retraining, re-sampling, or seed tuning is required

This design ensures that reported differences reflect **augmentation strategy effects**
rather than training noise or evaluation inconsistencies.

---

## Required Assets

Before running the reproduction scripts, ensure the following assets are available.

### 1. Model Checkpoints
Pretrained diffusion and Vision Transformer checkpoints:

# 📄 Reproducibility Guide for EEG-Neurodiffusion

This document describes how to **exactly reproduce all tables, figures, and results**
reported in the paper:

> **EEG-Neurodiffusion: Topology-Preserving Diffusion Models with Vision Transformers for Cognitive Load Classification**

⚠ **No retraining is required.**  
All results are generated using **frozen model checkpoints** and a **fixed real test split**.

---

## Reproducibility Philosophy

All experiments in this work follow a **frozen-checkpoint evaluation protocol** designed
to eliminate variability and ensure exact reproducibility.

Specifically:

- Each augmentation regime is trained **once**
- Model weights are saved as checkpoints
- All reported metrics, tables, and figures are generated from these fixed checkpoints
- A single evaluation pipeline is used for all comparisons
- No retraining, re-sampling, or seed tuning is required

This design ensures that reported differences reflect **augmentation strategy effects**
rather than training noise or evaluation inconsistencies.

---

## Required Assets

Before running the reproduction scripts, ensure the following assets are available.

### 1. Model Checkpoints
Pretrained diffusion and Vision Transformer checkpoints:

checkpoints/
├── diffusion/
│ ├── real_only/
│ ├── raw_diffusion/
│ ├── selective/
│ └── j2/
└── vit/

### 2. Preprocessed EEG Topomaps
Preprocessed EEG scalp topomaps with a fixed subject-independent split:


datasets/
├── train/
├── test_real/
└── metadata.csv

⚠ Synthetic samples are generated **only from training data**.

### 3. Classification Reports
JSON classification reports generated during evaluation:


⚠ Synthetic samples are generated **only from training data**.

### 3. Classification Reports
JSON classification reports generated during evaluation:

results/
├── real_only/
├── raw_diffusion/
├── selective/
└── j2/

Each folder contains per-fold and aggregated prediction outputs.

---

## Reproducing Paper Tables

All table-generation scripts are located in the `paper/` directory.

### Table 1 — Ablation Performance


python journal_stats_from_classification_report.py
Generates:

results_folds.csv

significance_tests.csv

bootstrap_ci.csv

LaTeX table: paper/tables/ablation_table.tex


### Table 2 — Degradation Analysis
python journal_degradation_analysis.py

Generates:

degradation_table.csv

LaTeX table: paper/tables/degradation_table.tex

