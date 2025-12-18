# EEG-Neurodiffusion

Diffusion-based generative modeling framework for EEG topographic representations and EEG-derived images, with applications in data augmentation, cross-subject generalization, and cognitive state classification.

---

## Overview

Electroencephalography (EEG) data is characterized by low signal-to-noise ratio, non-stationarity, and strong spatial–temporal dependencies. While GAN-based approaches have been explored for EEG data augmentation, they often suffer from mode collapse and unstable training.

**EEG-Neurodiffusion** explores *diffusion probabilistic models* as a principled alternative for high-fidelity EEG generation, particularly focusing on:

- EEG topographic maps (scalp projections)
- Class-conditional and task-conditional generation
- Robust augmentation for downstream Vision Transformer (ViT) classifiers

This repository provides the **complete research codebase** used for diffusion-based EEG synthesis and evaluation.

---

## Key Features

- Diffusion models (DDPM-style) for EEG topomap generation  
- U-Net–based diffusion backbone  
- Class-conditional and dataset-conditional generation  
- Integration with EEG topomap pipelines  
- Evaluation using statistical similarity metrics (MMD, PSD, SSIM)  
- Downstream evaluation using ViT classifiers  
- Designed for reproducible research workflows  

---

## Repository Structure

