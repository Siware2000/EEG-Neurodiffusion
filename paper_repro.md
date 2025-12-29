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

### Table 3 — Effect Size Analysis
python journal_effect_size_tables.py

Generates:

effect_size_table.csv

LaTeX table: paper/tables/effect_size_table.tex
### Table 4 — Topology Preservation Metrics

python run_topology_eval.py
python summarize_topology.py

Generates:

topology_metrics.csv

LaTeX tables:

paper/tables/topology_summary.tex

paper/tables/topology_by_class.tex
### Reproducing Figures
All figures are saved under:

paper/journal_figs/
### Confusion Matrices
python eval_confusion_matrices.py
Generates normalized confusion matrices for all augmentation regimes.

### ROC / AUC Curves
python eval_roc_curves.py
Generates one-vs-rest ROC curves and macro-average AUC plots.

### Vision Transformer Attention Rollout
python eval_attention_rollout.py
Generates ViT attention rollout overlays for qualitative inspection.

### Data Integrity & Leakage Control
The following safeguards are strictly enforced throughout the pipeline:

Subject-independent train / test splits

Synthetic samples generated only from training subjects

No validation or test data used during diffusion training

Evaluation performed only on real test samples

Synthetic data is never mixed with evaluation data

Identical evaluation code and thresholds across all regimes
### Notes for Reviewers and Editors
Diffusion augmentation is not assumed to improve performance

Results are framed in terms of robustness and degradation control

Confidence intervals are reported using bootstrap resampling

Effect sizes are reported to quantify magnitude of change

All scripts are deterministic given fixed checkpoints

Reported trends are reproducible without retraining

### Summary
This repository enables exact reproduction of all reported results and figures
using fixed checkpoints and deterministic evaluation scripts.

The reproducibility design ensures that conclusions reflect true augmentation behavior
rather than experimental noise.
