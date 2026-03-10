"""
evaluate.py
-----------
Evaluation metrics, plots, and the experiment matrix runner.

Functions
---------
compute_metrics     : AUROC, F1, Precision, Recall, Confusion Matrix
plot_roc_curve      : ROC curve for one detector/feature combination
plot_confusion_matrix
plot_score_distribution : distribution of anomaly scores (normal vs defective)
run_experiment_matrix   : full Feature × Detector grid → DataFrame of AUROC
save_results            : persist results as CSV
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, scores: np.ndarray,
                    y_pred: np.ndarray = None) -> dict:
    """
    Compute evaluation metrics.

    Parameters
    ----------
    y_true  : ground-truth labels  (0 = normal, 1 = defective)
    scores  : continuous anomaly score (higher = more anomalous)
    y_pred  : optional binary predictions; if None, derived from scores
              using the 95th-percentile threshold.

    Returns
    -------
    dict with keys: auroc, f1, precision, recall, confusion_matrix
    """
    auroc = roc_auc_score(y_true, scores)

    if y_pred is None:
        thr = np.percentile(scores, 95)
        y_pred = (scores >= thr).astype(int)

    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    return dict(auroc=auroc, f1=f1, precision=prec, recall=rec,
                confusion_matrix=cm)


# ────────────────────────────────────────────────────────────────────────────
# Plots
# ────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray,
                  label: str = "", out_path: str = None):
    """Plot and optionally save a single ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"{label}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_roc_curves_comparison(results: dict, out_path: str = None):
    """
    Plot multiple ROC curves on the same axes.

    Parameters
    ----------
    results : dict  {label_string: (y_true, scores)}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (y_true, scores) in results.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, lw=1.5, label=f"{label}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Method Comparison")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "", out_path: str = None):
    """Normalised confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Normal", "Defective"])
    disp.plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_score_distribution(scores_normal: np.ndarray,
                            scores_defect: np.ndarray,
                            title: str = "", out_path: str = None):
    """Histogram of anomaly scores for normal vs defective samples."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(scores_normal, bins=40, alpha=0.6, label="Normal", color="steelblue",
            density=True)
    ax.hist(scores_defect, bins=40, alpha=0.6, label="Defective", color="tomato",
            density=True)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(title or "Score Distribution")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_experiment_heatmap(df: pd.DataFrame, out_path: str = None):
    """
    Heatmap of the AUROC experiment matrix.

    Parameters
    ----------
    df : DataFrame with features as index and detectors as columns.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 1.2),
                                    max(4, len(df.index) * 0.6)))
    im = ax.imshow(df.values.astype(float), vmin=0.5, vmax=1.0, cmap="RdYlGn",
                   aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_title("AUROC — Feature × Detector Matrix")

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            if not np.isnan(float(val)):
                ax.text(j, i, f"{float(val):.3f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


# ────────────────────────────────────────────────────────────────────────────
# Interpretability / feature visualisation
# ────────────────────────────────────────────────────────────────────────────

def plot_pca_scatter(X_train: np.ndarray, X_test: np.ndarray,
                     y_test: np.ndarray, title: str = "",
                     out_path: str = None):
    """
    2D PCA scatter: projects features into the first two principal components
    and shows normal (blue) vs defective (red) clusters.

    Useful for oral exam: demonstrates separability of normal vs anomalous
    samples in the learned feature space.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_all = np.vstack([X_train, X_test])
    X_all_s = scaler.fit_transform(X_all)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_all_s)

    X_train_2d = X_2d[:len(X_train)]
    X_test_2d = X_2d[len(X_train):]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
               c="steelblue", alpha=0.4, s=20, label="Train (normal)")
    mask_n = y_test == 0
    mask_d = y_test == 1
    ax.scatter(X_test_2d[mask_n, 0], X_test_2d[mask_n, 1],
               c="mediumseagreen", alpha=0.6, s=25, label="Test (normal)")
    ax.scatter(X_test_2d[mask_d, 0], X_test_2d[mask_d, 1],
               c="tomato", alpha=0.7, s=30, marker="x", label="Test (defective)")

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)")
    ax.set_title(title or "PCA — Feature Space Projection")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_feature_importance(feature_names: list, importances: np.ndarray,
                            title: str = "", top_k: int = 20,
                            out_path: str = None):
    """
    Horizontal bar chart of feature importances (e.g. from Random Forest).

    Aids interpretability: shows which texture descriptors contribute most
    to the anomaly detection decision.
    """
    idx = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(6, max(3, top_k * 0.25)))
    ax.barh(range(len(idx)), importances[idx], color="steelblue")
    if feature_names:
        labels = [feature_names[i] if i < len(feature_names)
                  else f"f{i}" for i in idx]
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Importance")
    ax.set_title(title or "Feature Importance")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_preprocessing_comparison(img_raw: np.ndarray,
                                  img_processed: np.ndarray,
                                  title: str = "",
                                  out_path: str = None):
    """
    Side-by-side comparison of raw vs preprocessed images.
    Useful for showing denoising and contrast enhancement effects.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].imshow(img_raw, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(img_processed, cmap="gray")
    axes[1].set_title("Preprocessed")
    axes[1].axis("off")
    fig.suptitle(title or "Preprocessing Effect", fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


# ────────────────────────────────────────────────────────────────────────────
# Experiment matrix
# ────────────────────────────────────────────────────────────────────────────

def run_experiment_matrix(X_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          feature_sets: dict,
                          detector_classes: dict,
                          out_dir: str = "results") -> pd.DataFrame:
    """
    Run the full Feature × Detector experiment matrix.

    Parameters
    ----------
    X_train         : (N_train, D) feature matrix — normal samples only
    X_test          : (N_test,  D) feature matrix
    y_test          : (N_test,)    labels (0=normal, 1=defective)
    feature_sets    : dict {name: column_indices_or_slice}
                      Use None to mean "all features".
    detector_classes: dict {name: DetectorClass}

    Returns
    -------
    DataFrame of AUROC values — rows=features, columns=detectors
    """
    os.makedirs(out_dir, exist_ok=True)
    records = {}
    all_scores = {}  # label -> (y_true, scores) for top-N ROC comparison

    for feat_name, feat_idx in feature_sets.items():
        Xtr = X_train[:, feat_idx] if feat_idx is not None else X_train
        Xte = X_test[:, feat_idx]  if feat_idx is not None else X_test
        row = {}

        for det_name, DetClass in detector_classes.items():
            print(f"  [{feat_name}] + [{det_name}] ...", end=" ", flush=True)
            try:
                det = DetClass()
                det.fit(Xtr)
                scores = det.score_samples(Xte)
                auroc = roc_auc_score(y_test, scores)
                print(f"AUROC={auroc:.4f}")
                all_scores[f"{feat_name}+{det_name}"] = (y_test, scores)

            except Exception as e:
                print(f"ERROR: {e}")
                auroc = float("nan")

            row[det_name] = round(auroc, 4) if not np.isnan(auroc) else np.nan

        records[feat_name] = row

    df = pd.DataFrame(records).T
    df.index.name = "Features"

    # Save a single ROC comparison with the top-5 combinations by AUROC
    if all_scores:
        top5 = sorted(
            all_scores.items(),
            key=lambda kv: roc_auc_score(kv[1][0], kv[1][1]),
            reverse=True
        )[:5]
        plot_roc_curves_comparison(
            {label: pair for label, pair in top5},
            out_path=os.path.join(out_dir, "roc_top5_comparison.png")
        )

    return df


# ────────────────────────────────────────────────────────────────────────────
# Save / load
# ────────────────────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame, out_dir: str = "results",
                 prefix: str = "matrix"):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{prefix}_auroc.csv")
    df.to_csv(csv_path)
    print(f"Results saved to {csv_path}")

    heatmap_path = os.path.join(out_dir, f"{prefix}_heatmap.png")
    plot_experiment_heatmap(df, out_path=heatmap_path)
    print(f"Heatmap saved to {heatmap_path}")
