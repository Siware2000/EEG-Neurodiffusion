
---

# 📄 `paper_repro.md`

```markdown
# Reproducibility Guide for EEG-Neurodiffusion

This document describes how to reproduce all tables, figures, and results reported in the paper:

**EEG-Neurodiffusion: Topology-Preserving Diffusion Models with Vision Transformers for Cognitive Load Classification**

No retraining is required.

---

## Reproducibility Philosophy

All experimental results in the paper are generated using a frozen-checkpoint evaluation protocol:

- Each augmentation regime is trained once
- Model weights are saved
- All reported metrics, tables, and figures are generated from these fixed checkpoints
- A single evaluation script is used to ensure consistency

This design eliminates variability due to random seeds, resampling, or retraining.

---

## Required Assets

Before running the scripts, ensure the following are present:

- Pretrained diffusion and ViT checkpoints (`checkpoints/`)
- Preprocessed EEG topomaps (training and fixed real test split)
- Classification report JSON files generated during evaluation

---

## Reproducing Paper Tables

### Table: Ablation Performance

```bash
python journal_stats_from_classification_report.py

Generates:

results_folds.csv

significance_tests.csv

bootstrap_ci.csv

Table: Degradation Analysis
python journal_degradation_analysis.py


Generates:

degradation_table.csv

LaTeX table: tables/degradation_table.tex

Table: Effect Size Analysis
python journal_effect_size_tables.py


Generates:

effect_size_table.csv

LaTeX table: tables/effect_size_table.tex

Reproducing Figures
Confusion Matrices
python eval_confusion_matrices.py

ROC/AUC Curves
python eval_roc_curves.py

ViT Attention Rollout Visualizations
python eval_attention_rollout.py


All figures are saved to:

paper/journal_figs/

Data Integrity and Leakage Control

All datasets use subject-independent splits

Synthetic samples are generated exclusively from training data

No validation or test subjects are used during diffusion training

Evaluation is performed only on real test samples

Notes for Reviewers and Editors

Diffusion augmentation is not assumed to improve performance

Results are framed in terms of degradation control and robustness

Confidence intervals are reported using bootstrap resampling

All scripts are deterministic given fixed checkpoints
