# 📄 Reproducibility Guide for EEG-Neurodiffusion

This document describes how to reproduce **all tables, figures, and results** reported in the paper:

> **EEG-Neurodiffusion: Topology-Preserving Diffusion Models with Vision Transformers for Cognitive Load Classification**

⚠ **No retraining is required.**

---

## Reproducibility Philosophy

All experimental results in the paper are generated using a **frozen-checkpoint evaluation protocol**:

- Each augmentation regime is trained **once**
- Model weights are saved as checkpoints
- All reported metrics, tables, and figures are generated from these fixed checkpoints
- A single evaluation pipeline is used to ensure consistency

This design eliminates variability due to random seeds, resampling, or retraining and enables exact reproducibility.

---

## Required Assets

Before running the reproduction scripts, ensure the following assets are present:

- Pretrained diffusion and ViT checkpoints

checkpoints/
- Preprocessed EEG topomaps (training + fixed real test split)  
datasets/
- Classification report files in JSON format generated during evaluation  
results/
├── real_only/
├── raw_diffusion/
├── selective/
└── j2/

---

## Reproducing Paper Tables

All table-generation scripts are located in the `paper/` directory.

### Table 1: Ablation Performance

```bash
python journal_stats_from_classification_report.py
Generates:

results_folds.csv

significance_tests.csv

bootstrap_ci.csv

LaTeX table: paper/tables/ablation_table.tex
Table 2: Degradation Analysis
python journal_degradation_analysis.py

Generates:

degradation_table.csv
LaTeX table: paper/tables/degradation_table.tex
Table 3: Effect Size Analysis
python journal_effect_size_tables.py


Generates:

effect_size_table.csv

LaTeX table: paper/tables/effect_size_table.tex

Reproducing Figures

All figures are saved under:

paper/journal_figs/

Confusion Matrices
python eval_confusion_matrices.py


Generates normalized confusion matrices for all augmentation regimes.

ROC / AUC Curves
python eval_roc_curves.py


Generates one-vs-rest ROC curves and macro-average AUC plots.

ViT Attention Rollout Visualizations
python eval_attention_rollout.py


Generates Vision Transformer attention rollout overlays for qualitative inspection.

Data Integrity and Leakage Control
The following safeguards are enforced throughout the pipeline:
All datasets use subject-independent splits
Synthetic samples are generated only from training data
No validation or test subjects are used during diffusion training
Evaluation is performed only on real test samples
Synthetic data is never mixed with evaluation data

Notes for Reviewers and Editors

Diffusion augmentation is not assumed to improve performance
Results are framed in terms of performance degradation control and robustness
Confidence intervals are reported using bootstrap resampling
Effect sizes are reported to quantify the magnitude of change
All scripts are deterministic given fixed checkpoints
