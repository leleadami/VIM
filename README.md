# Interpretable Texture-Based Anomaly Detection for Industrial Surface Inspection

Classical signal processing and machine learning pipeline for anomaly detection on industrial surfaces. **No deep learning, no pretrained neural networks** — every method is fully interpretable and explainable.

Final exam project (6 ECTS) — University of Trento.

## Overview

The system detects surface defects on industrial products using the [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). It follows a one-class classification paradigm: models are trained exclusively on normal (defect-free) images, then anomalies are detected as deviations from the learned normal distribution.

**Best result:** Stats + Isolation Forest on *wood* — **AUROC 0.9316**

## Pipeline Architecture

```
MVTec images
    │
    ▼
┌─────────────────────────────────┐
│  1. PREPROCESSING               │
│  Denoising: Gaussian, Median,   │
│  Bilateral, NLM, Wavelet, RCLBP │
│  Contrast: CLAHE, Hist. Eq.     │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  2. FEATURE EXTRACTION          │
│  LBP, CLBP, Gabor, GLCM, HOG, │
│  FFT, Wavelet, Stats, Laws,    │
│  Dense SIFT + BoVW             │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  3. DIMENSIONALITY REDUCTION    │
│  PCA (95% variance)            │
│  + PCA Null Subspace scoring   │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  4. ANOMALY DETECTION           │
│  OC-SVM, Isolation Forest, LOF, │
│  GMM, Elliptic Envelope, KDE,  │
│  kNN, Mahalanobis, Ensemble    │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  5. EVALUATION                  │
│  AUROC, F1, Precision, Recall, │
│  ROC curves, Heatmaps          │
└─────────────────────────────────┘
```

## Project Structure

```
ESI/
├── pipeline.py            # Experiment matrix runner (features × detectors)
├── predict_model.py       # Single-image predictor with graphical output
├── requirements.txt
├── src/
│   ├── dataset.py             # MVTec AD loader
│   ├── preprocessing.py       # 6 denoisers + CLAHE/HE
│   ├── feature_extraction.py  # 11 feature extractors
│   ├── models.py              # 10 detectors + PCA Null Subspace
│   └── evaluate.py            # Metrics and plots
├── docs/
│   ├── esame.tex              # Compact two-column exam report
│   ├── report.tex             # Exhaustive technical reference (60+ pages)
│   ├── codice.tex             # Code documentation
│   ├── comandi.md             # Usage commands
│   ├── figures/               # LaTeX figures
│   └── papers/                # References (.bib + PDFs)
├── dataset/                   # MVTec AD (not tracked in git)
├── models/                    # Trained .pkl models
└── results/                   # CSV tables, ROC curves, heatmaps
```

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd ESI

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, NumPy, SciPy, OpenCV, scikit-learn, scikit-image, Pandas, Matplotlib, PyWavelets.

## Dataset

Download the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place it in `dataset/`. Expected structure:

```
dataset/
├── wood/
│   ├── train/good/        # Normal images (training)
│   └── test/
│       ├── good/          # Normal images (test)
│       ├── color/         # Defect type 1
│       ├── hole/          # Defect type 2
│       └── scratch/       # Defect type 3
├── tile/
├── grid/
├── hazelnut/
├── carpet/
└── leather/
```

## Usage

### 1. Run the experiment pipeline

The pipeline evaluates all combinations of feature descriptors and anomaly detectors, producing AUROC scores, ROC curves, and heatmaps.

```bash
# Single category
python pipeline.py --dataset dataset --category wood

# All categories sequentially
python pipeline.py --dataset dataset --all-categories

# Specific features and detectors
python pipeline.py --dataset dataset --category tile \
    --features lbp glcm gabor --detectors OC-SVM IsolationForest
```

Results are saved to `results/<category>/`:
- `<category>_unsupervised_auroc.csv` — AUROC matrix (features × detectors)
- `<category>_unsupervised_heatmap.png` — Visual heatmap of results
- `roc_top5_comparison.png` — ROC curves for top 5 combinations
- `pca_scatter_all.png` — PCA scatter plot
- `preprocessing_comparison.png` — Denoising comparison

### 2. Train best models for prediction

After running the pipeline, train the best-performing model for each category:

```bash
# Train all categories (reads best combo from pipeline CSV)
python predict_model.py --train-all

# Train a single category
python predict_model.py --category wood --retrain
```

Models are saved to `models/<category>_best.pkl`.

### 3. Predict on a single image

```bash
# Auto-detects category from path
python predict_model.py dataset/wood/test/scratch/001.png

# Explicit category
python predict_model.py image.png --category wood
```

Outputs a graphical panel showing the image, anomaly score, classification (NORMALE/DIFETTOSO), and score distribution.

## Feature Descriptors

| Feature | Description |
|---|---|
| **LBP** | Local Binary Pattern — texture microstructure |
| **LBP Multi-Scale** | LBP at 3 scales (R=1,2,3) concatenated |
| **CLBP** | Completed LBP with sign + magnitude components |
| **Gabor** | Filter bank (6 orientations × 4 scales) — frequency analysis |
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

## Key References

- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019
- Gyimah et al., *RCLBP: Robust Completed Local Binary Pattern for Surface Defect Detection*, arXiv:2112.04021
- Laws, *Textured Image Segmentation*, PhD thesis, USC, 1980
- Scholkopf et al., *Estimating the Support of a High-Dimensional Distribution*, Neural Computation, 2001

## License

MIT License — see [LICENSE](LICENSE).
