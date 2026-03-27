# Interpretable Texture-Based Anomaly Detection for Industrial Surface Inspection

Classical signal processing and machine learning pipeline for anomaly detection on industrial surfaces.
**No deep learning, no pretrained neural networks** — every method is fully interpretable and explainable.

> Final exam project (6 ECTS) — University of Trento

## Overview

The system detects surface defects on industrial products using the [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). It follows a **one-class classification** paradigm: models are trained exclusively on normal (defect-free) images, then anomalies are detected as deviations from the learned normal distribution.

**Best result:** Stats + Isolation Forest on *wood* — **AUROC 0.9316**

## Getting Started

### Prerequisites

- Python 3.9+
- [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) placed in `dataset/`

### Installation

```bash
git clone <repo-url> && cd ESI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the MVTec AD dataset and place it in `dataset/` with this structure:

```
dataset/
├── wood/
│   ├── train/good/          # Normal images (training)
│   └── test/
│       ├── good/            # Normal images (test)
│       ├── color/           # Defect type
│       ├── hole/
│       └── scratch/
├── tile/
├── grid/
├── hazelnut/
└── ...
```

## Usage

The project provides two entry points: `pipeline.py` for running experiments and `predict_model.py` for single-image inference.

---

### `pipeline.py` — Experiment Runner

Evaluates all combinations of feature descriptors and anomaly detectors, producing AUROC scores, ROC curves, and heatmaps.

```
usage: pipeline.py [-h] --dataset PATH [--category NAME] [--all-categories]
                   [--features F [F ...]] [--detectors D [D ...]]
                   [--out DIR] [--img-size {128,256,512,1024}]
                   [--denoise {gaussian,median,bilateral,nlmeans,wavelet,rclbp,none}]
                   [--enhance {clahe,histeq,none}]
```

| Option | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Path to MVTec root directory |
| `--category` | — | Single category (e.g. `wood`, `tile`, `grid`) |
| `--all-categories` | — | Run on all categories found in `--dataset` |
| `--features` | all | Feature groups to evaluate (see table below) |
| `--detectors` | all | Detectors to evaluate (see table below) |
| `--out` | `results/` | Output directory |
| `--img-size` | `256` | Resize images to NxN |
| `--denoise` | `gaussian` | Denoising filter |
| `--enhance` | `clahe` | Contrast enhancement |

**Examples:**

```bash
# Run full experiment matrix on a single category
python pipeline.py --dataset dataset --category wood

# Run on all categories
python pipeline.py --dataset dataset --all-categories

# Select specific features and detectors
python pipeline.py --dataset dataset --category tile \
    --features Stats HOG --detectors OC-SVM IsolationForest

# Custom preprocessing and image size
python pipeline.py --dataset dataset --category grid \
    --denoise bilateral --enhance histeq --img-size 512

```

**Output** (`results/<category>/`):

| File | Content |
|---|---|
| `*_unsupervised_auroc.csv` | AUROC matrix (features x detectors) |
| `*_unsupervised_heatmap.png` | Visual heatmap of results |
| `roc_top5_comparison.png` | ROC curves for top 5 combinations |
| `pca_scatter_all.png` | PCA scatter plot |
| `preprocessing_comparison.png` | Before/after denoising |

---

### `predict_model.py` — Single-Image Predictor

Classifies a surface image as normal or defective using a trained model. On first run, the model is automatically trained from the dataset and cached to disk.

```
usage: predict_model.py [-h] [--category {wood,tile,grid,hazelnut,carpet,leather}]
                        [--retrain] [--train-all]
                        [image]
```

| Option | Default | Description |
|---|---|---|
| `image` | — | Path to the image to analyse |
| `--category` | auto | MVTec category (auto-detected from path if omitted) |
| `--retrain` | — | Force retraining even if a cached model exists |
| `--train-all` | — | Train and save the best model for every category |

**Examples:**

```bash
# Predict on a single image (category auto-detected from path)
python predict_model.py dataset/wood/test/scratch/001.png

# Explicit category
python predict_model.py image.png --category wood

# Force retraining before prediction
python predict_model.py dataset/tile/test/rough/003.png --retrain

# Train all models at once (reads best combo from pipeline CSV)
python predict_model.py --train-all
```

Models are cached in `models/<category>_best.pkl`.

## Feature Descriptors

| Feature | Description |
|---|---|
| **LBP** | Local Binary Pattern — texture microstructure |
| **LBP Multi-Scale** | LBP at 3 scales (R=1,2,3) concatenated |
| **CLBP** | Completed LBP with sign + magnitude components |
| **Gabor** | Filter bank (6 orientations x 4 scales) — frequency analysis |
| **GLCM** | Gray-Level Co-occurrence Matrix — statistical texture |
| **HOG** | Histogram of Oriented Gradients — edge structure |
| **FFT** | Frequency band energy via 2D Fourier Transform |
| **Wavelet** | Sub-band energy via discrete wavelet transform |
| **Stats** | Statistical moments (mean, std, skewness, kurtosis) |
| **Laws** | Laws Texture Energy (L5/E5/S5/R5 kernels) |
| **Dense SIFT + BoVW** | Bag of Visual Words with dense SIFT descriptors |

## Anomaly Detectors

| Detector | Type | Description |
|---|---|---|
| **OC-SVM** | One-class | Support Vector Machine with RBF kernel |
| **Isolation Forest** | One-class | Ensemble of random trees isolating anomalies |
| **LOF** | One-class | Local Outlier Factor — density-based |
| **GMM** | One-class | Gaussian Mixture Model + Mahalanobis distance |
| **Elliptic Envelope** | One-class | Robust covariance estimation |
| **KDE** | One-class | Kernel Density Estimation |
| **kNN** | One-class | k-Nearest Neighbor distance |
| **Mahalanobis** | One-class | Mahalanobis distance with Ledoit-Wolf covariance |
| **Ensemble** | One-class | Score fusion of IF + OC-SVM + LOF + EE |
| **PCA Null Subspace** | Unsupervised | Projection onto low-variance subspace |

## Project Structure

```
ESI/
├── pipeline.py              # Experiment matrix runner (features x detectors)
├── predict_model.py         # Single-image predictor with graphical output
├── predict_models_all.py    # Batch prediction across categories
├── requirements.txt
├── src/
│   ├── dataset.py           # MVTec AD loader
│   ├── preprocessing.py     # 6 denoisers + CLAHE / Histogram Equalization
│   ├── feature_extraction.py# 11 feature extractors
│   ├── models.py            # 10 detectors + PCA Null Subspace
│   └── evaluate.py          # Metrics, ROC curves, heatmaps
├── dataset/                 # MVTec AD images (not tracked)
├── models/                  # Trained .pkl models (not tracked)
└── results/                 # CSV tables, ROC curves, heatmaps (not tracked)
```

## References

- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019
- Gyimah et al., *RCLBP: Robust Completed Local Binary Pattern for Surface Defect Detection*, arXiv:2112.04021
- Laws, *Textured Image Segmentation*, PhD thesis, USC, 1980
- Scholkopf et al., *Estimating the Support of a High-Dimensional Distribution*, Neural Computation, 2001

## License

MIT License — see [LICENSE](LICENSE).
