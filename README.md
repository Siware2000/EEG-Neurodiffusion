EEG-Neurodiffusion

Topology-Preserving Diffusion Models for EEG Topomaps and Cognitive Load Classification

EEG-Neurodiffusion is a research-grade framework for diffusion-based generative modeling of EEG scalp topographic maps (topomaps), with a focus on topology preservation, controlled data augmentation, and robust cognitive load classification under subject-independent evaluation.

This repository contains the complete, reproducible codebase used in the paper:

EEG-Neurodiffusion: Topology-Preserving Diffusion Models with Vision Transformers for Cognitive Load Classification

Motivation

Electroencephalography (EEG) signals are characterized by:

Low signal-to-noise ratio

Strong inter-subject variability

Spatial correlations induced by volume conduction and cortical geometry

Limited labeled data and class imbalance

While generative models are often used to augment EEG data, uncontrolled synthetic injection can distort spatial relationships between electrodes, introducing samples that are statistically plausible yet physiologically implausible.

EEG-Neurodiffusion studies diffusion models not as a guaranteed performance booster, but as a risk-sensitive augmentation mechanism whose benefits depend critically on topology preservation and quality control.

Core Idea

We explore denoising diffusion probabilistic models (DDPMs) trained on EEG scalp topomaps, combined with a Vision Transformer (ViT) classifier, and systematically analyze:

When diffusion helps

When diffusion hurts

Why naive augmentation fails

How topology-aware quality control mitigates negative transfer

Rather than reporting only peak accuracy, this work emphasizes failure-mode analysis under frozen, reproducible evaluation.

Key Contributions

Topology-preserving EEG diffusion modeling using conditional DDPMs

Multi-band EEG topomaps encoded as image channels

U-Net diffusion backbone with class conditioning

Synthetic data quality control strategies:

Raw diffusion augmentation

Selective (class-aware) diffusion injection

Classifier-guided filtering (J2)

Topology preservation metrics:

Radial PSD correlation

Electrode-distance Mantel correlation

Direct topology preservation score

Frozen-checkpoint ablation protocol for fair comparison

Vision Transformer (ViT) as a topology-sensitive downstream classifier

Journal-ready statistical testing, degradation analysis, and interpretability

Repository Structure
EEG-Neurodiffusion/
│
├── src/                     # Core modeling & evaluation code
│   ├── diffusion/            # DDPM + U-Net implementation
│   ├── vit/                  # Vision Transformer classifier
│   ├── metrics/              # MMD, PSD similarity, topology metrics
│   └── scripts/              # Training, evaluation, analysis scripts
│
├── paper/
│   ├── tables/               # LaTeX-ready tables
│   ├── journal_outputs/      # CSV outputs from analysis scripts
│   ├── journal_figs/         # Confusion matrices, ROC, attention maps
│   └── main.tex              # Journal manuscript
│
├── checkpoints/              # Saved diffusion and ViT checkpoints
├── datasets/                 # Preprocessed EEG topomaps (not included)
├── requirements.txt
└── README.md

Reproducibility & Paper Mapping

This repository is structured to exactly reproduce all reported tables and figures without retraining.

Paper Component	Script
Ablation Table	journal_stats_from_classification_report.py
Degradation Analysis	journal_degradation_analysis.py
Effect Size Tables	journal_effect_size_tables.py
Statistical Testing	journal_stats_from_classification_report.py
Confusion Matrices	eval_confusion_matrices.py
ROC / AUC Curves	eval_roc_curves.py
ViT Attention Rollout	eval_attention_rollout.py
Topology Metrics	run_topology_eval.py, summarize_topology.py

All results are generated using frozen checkpoints and a fixed real test split.

Evaluation Protocol (Frozen-Checkpoint)

To isolate the effect of diffusion-based augmentation:

Each augmentation regime is trained once

All evaluations reuse saved checkpoints

No retraining is required to reproduce results

Subject-independent splits are strictly enforced

Synthetic samples are generated only from training data

No validation or test leakage occurs

This ensures fair, stable, and reproducible comparisons.

Key Findings (High-Level)

Naive diffusion augmentation can significantly degrade performance

Spectral similarity alone does not guarantee spatial consistency

Diffusion models can generate visually plausible but topologically distorted EEG samples

Selective, class-aware diffusion provides the best robustness–performance trade-off

Confidence-only filtering (J2) is conservative but incomplete

Diffusion augmentation should be treated as damage control, not guaranteed improvement

Neurophysiological Perspective

From a neurophysiological standpoint, EEG electrodes measure correlated activity from overlapping cortical sources due to volume conduction and anatomical proximity. Preserving local electrode neighborhood structure is therefore critical.

Synthetic topomaps that match global spectral statistics but violate these spatial dependencies may encode physiologically implausible activation patterns, explaining their negative impact on downstream classification despite high visual fidelity.

Neurophysiological Perspective

From a neurophysiological standpoint, EEG electrodes measure correlated activity from overlapping cortical sources due to volume conduction and anatomical proximity. Preserving local electrode neighborhood structure is therefore critical.

Synthetic topomaps that match global spectral statistics but violate these spatial dependencies may encode physiologically implausible activation patterns, explaining their negative impact on downstream classification despite high visual fidelity.
