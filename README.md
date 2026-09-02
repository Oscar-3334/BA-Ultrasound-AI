# BA-Ultrasound-AI

Source code for three-class biliary atresia (BA) diagnosis from ultrasound images.

## Overview

This repository contains the source code used for the three-class BA diagnostic experiments reported in the associated manuscript.

The code implements the data processing, model training, validation, testing, and evaluation procedures used in the study.

## Method

The framework is designed for automated diagnosis of:

- Biliary atresia (BA)
- Cholestasis
- Normal controls

The implementation is based on PyTorch.

## Code Structure

- `dataset.py` — Dataset loading and preprocessing
- `model.py` — Model architecture
- `loss.py` — Loss functions
- `engine.py` — Training and evaluation procedures
- `utils.py` — Utility functions and evaluation metrics
- `main.py` — Main training script
- `val_only.py` — Validation procedures
- `test_only.py` — Test procedures
- `enhanced_evaluation.py` — Detailed evaluation and visualization
- `gradcam_3cls.py` — Grad-CAM analysis for three-class diagnosis
- `uq_metrics.py` — Uncertainty-related evaluation metrics

## Data Availability

The ultrasound images used in this study are not included in this repository because the data are subject to privacy and ethical restrictions.

Users should provide their own authorized dataset and configure the corresponding data paths before running the code.

## Requirements

The code was implemented in Python using the PyTorch framework.

Required Python packages include:

- PyTorch
- torchvision
- NumPy
- pandas
- scikit-learn
- SciPy
- OpenCV
- Pillow
- matplotlib
- seaborn
- tqdm

Detailed installation and configuration instructions will be provided in future updates.

## Usage

Detailed instructions for dataset preparation, training, validation, testing, and model evaluation will be added in future updates.

## Model Weights

Pre-trained and trained model weights will be described and provided separately where applicable.

## License

License information will be added upon the public release of the repository.
