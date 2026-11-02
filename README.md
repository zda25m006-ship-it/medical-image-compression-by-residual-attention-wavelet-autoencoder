# RAWA: Residual Attention Wavelet Autoencoder

A PyTorch implementation for near-lossless brain MRI compression using a residual attention wavelet autoencoder. The repository includes both a full training pipeline and a separate evaluation script for re-running results on saved model weights.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

This project explores a hybrid image-compression approach that combines:

- Haar wavelet decomposition
- residual encoder-decoder blocks
- CBAM attention at the bottleneck
- multi-scale skip connections
- multi-term reconstruction loss (MSE + MS-SSIM + gradient loss)

The goal is to preserve clinically relevant anatomy and texture while achieving high compression ratios on grayscale brain MRI slices.

## Repository Structure

```text
.
├── LICENSE
├── README.md
├── compression_model.py      # full training pipeline
├── eval_only.py              # loads saved checkpoints and evaluates plots
├── results/
│   ├── comparison_table.csv  # generated metrics table
│   ├── metrics_comparison.png
│   ├── psnr_vs_cr.png
│   ├── reconstruction_comparison.png
│   ├── training_curves.png
│   └── samples/
├── .gitignore
└── ...
```

## Key Model Components

### RAWA architecture

- Haar DWT converts MRI images into LL/LH/HL/HH wavelet sub-bands.
- A convolutional encoder extracts multi-scale features.
- CBAM attention emphasizes clinically important regions and structural details.
- Multi-scale skip connections preserve anatomical information.
- Haar IDWT reconstructs the compressed image back to the spatial domain.

### Baseline comparison

The training pipeline also builds a plain CNN autoencoder baseline so the model can be compared directly against a conventional convolutional autoencoder.

## Installation

Install the required dependencies:

```bash
pip install torch torchvision opencv-python scikit-image pandas matplotlib pillow
```

## Dataset

The dataset is expected to be arranged as follows:

```text
d:\archive (2)\
├── Training\
│   ├── glioma\
│   ├── meningioma\
│   ├── notumor\
│   └── pituitary\
└── Testing\
    ├── glioma\
    ├── meningioma\
    ├── notumor\
    └── pituitary\
```

This project uses a Windows-style absolute path in the script (`d:\archive (2)`). If your dataset is stored elsewhere, update the `BASE_DIR` variable in the Python files.

## Training

Run the full training pipeline:

```bash
python compression_model.py
```

This script will:

- load the MRI dataset
- train the CNN baseline and RAWA model
- compute evaluation metrics
- generate comparison plots and reconstruction figures
- save model checkpoints in `results/`

## Evaluation Only

If you already have trained model weights saved under `results/`, run:

```bash
python eval_only.py
```

This script loads the saved weights and reproduces the evaluation and visualization workflow without retraining.

## Outputs

After training or evaluation, the project writes the following outputs to `results/`:

- `cnn_ae.pth`
- `rawa.pth`
- `comparison_table.csv`
- `metrics_comparison.png`
- `psnr_vs_cr.png`
- `training_curves.png`
- `reconstruction_comparison.png`
- `samples/`

## Training Configuration

The current implementation uses:

- Image size: 128 × 128 grayscale
- Batch size: 16
- Epochs: 25
- Optimizer: AdamW
- Learning rate: 3e-4
- Scheduler: Cosine Annealing
- Device: CUDA if available, otherwise CPU

## Loss Function

The RAWA model uses a composite loss designed to balance fidelity and perceptual quality:

```text
L = 0.50 × MSE + 0.35 × (1 − MS-SSIM) + 0.15 × Gradient-L1
```

This improves reconstruction quality while preserving edges and texture, which is especially important for medical imaging.

## Notes

- The project is designed for grayscale brain MRI images.
- Performance numbers are computed in the evaluation scripts based on the current dataset split.
- The code is optimized for CPU execution as well as GPU when available.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
