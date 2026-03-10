"""
pipeline.py
-----------
End-to-end runner for the experiment matrix.

Usage
-----
# Run all features × all unsupervised detectors on 'wood' category:
    python pipeline.py --dataset /path/to/mvtec --category wood

# Run a single feature + single detector:
    python pipeline.py --dataset /path/to/mvtec --category tile \
        --features lbp glcm --detectors OC-SVM IsolationForest

# Run all categories sequentially:
    python pipeline.py --dataset /path/to/mvtec --all-categories

# Results are written to results/<category>/
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

from dataset import load_mvtec_category, list_categories
from preprocessing import preprocess_batch
from feature_extraction import extract_batch, AVAILABLE_FEATURES
from models import UNSUPERVISED_DETECTORS, SUPERVISED_CLASSIFIERS
from evaluate import (
    run_experiment_matrix, save_results,
    plot_roc_curves_comparison, compute_metrics,
    plot_pca_scatter, plot_preprocessing_comparison
)


# ────────────────────────────────────────────────────────────────────────────
# Feature-set definitions for the experiment matrix
# ────────────────────────────────────────────────────────────────────────────
# Each entry: name → list of feature extractor names from AVAILABLE_FEATURES.
# "fused" uses all features concatenated.

FEATURE_GROUPS = {
    "LBP":              ["lbp"],
    "LBP_MS":           ["lbp_ms"],
    "CLBP":             ["clbp"],
    "Gabor":            ["gabor"],
    "GLCM":             ["glcm"],
    "HOG":              ["hog"],
    "FFT":              ["fft"],
    "Wavelet":          ["wavelet"],
    "Stats":            ["stats"],
    "LBP+GLCM+Gabor":   ["lbp", "glcm", "gabor"],
    "LBP_MS+GLCM+Gabor": ["lbp_ms", "glcm", "gabor"],
    "All":              AVAILABLE_FEATURES,
}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _get_feature_dim_ranges(feature_names_list: list) -> dict:
    """
    Build a dict {group_name: column_slice} so we can slice the full
    feature matrix by feature group.

    We extract a dummy vector for each group to know its length.
    """
    dummy = np.zeros((256, 256), dtype=np.uint8)
    slices = {}
    for group_name, group_features in feature_names_list:
        from feature_extraction import extract_features
        single = extract_features(dummy, group_features)
        slices[group_name] = len(single)
    return slices


def run_category(dataset_root: str, category: str,
                 feature_groups: dict,
                 detector_names: list,
                 out_root: str = "results",
                 img_size: tuple = (256, 256),
                 denoise: str = "gaussian",
                 enhance: str = "clahe"):
    """
    Full pipeline for a single MVTec category.

    1. Load images
    2. Preprocess
    3. Extract features for each group
    4. Run experiment matrix (unsupervised)
    5. Run supervised baselines
    6. Save all results
    """
    out_dir = os.path.join(out_root, category)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Category: {category.upper()}")
    print(f"{'='*60}")

    # ── 1. Load ──────────────────────────────────────────────────────────────
    t0 = time.time()
    X_train_raw, y_train, X_test_raw, y_test, meta = load_mvtec_category(
        dataset_root, category, img_size=img_size
    )

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print(f"[{category}] Preprocessing...")
    X_train_pp = preprocess_batch(X_train_raw, denoise=denoise, enhance=enhance)
    X_test_pp  = preprocess_batch(X_test_raw,  denoise=denoise, enhance=enhance)

    # ── 2b. Save preprocessing comparison plot ──────────────────────────────
    if len(X_train_raw) > 0:
        plot_preprocessing_comparison(
            X_train_raw[0], X_train_pp[0],
            title=f"{category} — Preprocessing ({denoise or 'none'} + {enhance or 'none'})",
            out_path=os.path.join(out_dir, "preprocessing_comparison.png")
        )

    # ── 3. Feature extraction per group ──────────────────────────────────────
    # Build the full feature matrix (all features concatenated) once,
    # then slice it per group for the experiment matrix.

    print(f"[{category}] Extracting features (all groups)...")
    X_train_full = extract_batch(X_train_pp, feature_names=AVAILABLE_FEATURES)
    X_test_full  = extract_batch(X_test_pp,  feature_names=AVAILABLE_FEATURES)

    # Build column index ranges per group
    dummy = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
    col_start = 0
    feat_slices = {}
    for feat_name in AVAILABLE_FEATURES:
        from feature_extraction import extract_features
        n_cols = len(extract_features(dummy, [feat_name]))
        feat_slices[feat_name] = (col_start, col_start + n_cols)
        col_start += n_cols

    # Map group names to column arrays
    group_indices = {}
    for group_name, group_feats in feature_groups.items():
        cols = []
        for fn in group_feats:
            s, e = feat_slices[fn]
            cols.extend(range(s, e))
        group_indices[group_name] = cols

    # ── 4. Experiment matrix — unsupervised ───────────────────────────────────
    selected_detectors = {k: v for k, v in UNSUPERVISED_DETECTORS.items()
                          if k in detector_names or not detector_names}

    # Convert to slice-friendly format for run_experiment_matrix
    feat_index_map = {name: np.array(cols) for name, cols in group_indices.items()}

    # ── 3b. PCA scatter of full features ────────────────────────────────────
    plot_pca_scatter(
        X_train_full[y_train == 0], X_test_full, y_test,
        title=f"{category} — PCA Feature Space (all descriptors)",
        out_path=os.path.join(out_dir, "pca_scatter_all.png")
    )

    print(f"\n[{category}] Running unsupervised experiment matrix "
          f"({len(feat_index_map)} features × {len(selected_detectors)} detectors)...")

    df_unsup = run_experiment_matrix(
        X_train=X_train_full[y_train == 0],   # train on normal only
        X_test=X_test_full,
        y_test=y_test,
        feature_sets=feat_index_map,
        detector_classes=selected_detectors,
        out_dir=os.path.join(out_dir, "unsupervised"),
    )
    save_results(df_unsup, out_dir=os.path.join(out_dir, "unsupervised"),
                 prefix=f"{category}_unsupervised")

    print(f"\n[{category}] Unsupervised AUROC matrix:")
    print(df_unsup.to_string())

    # ── 5. Supervised baselines ───────────────────────────────────────────────
    if y_train.sum() == 0:
        # Normal-only training set — skip supervised
        print(f"\n[{category}] Skipping supervised (no defect labels in train).")
    else:
        print(f"\n[{category}] Training supervised classifiers...")
        sup_dir = os.path.join(out_dir, "supervised")
        os.makedirs(sup_dir, exist_ok=True)
        sup_results = {}

        for model_name, ModelClass in SUPERVISED_CLASSIFIERS.items():
            for feat_name, feat_cols in feat_index_map.items():
                key = f"{feat_name}+{model_name}"
                Xtr = X_train_full[:, feat_cols]
                Xte = X_test_full[:, feat_cols]
                try:
                    clf = ModelClass()
                    clf.fit(Xtr, y_train)
                    proba = clf.predict_proba(Xte)
                    from sklearn.metrics import roc_auc_score
                    auroc = roc_auc_score(y_test, proba)
                    print(f"  {key}: AUROC={auroc:.4f}")
                    sup_results[key] = auroc
                except Exception as e:
                    print(f"  {key}: ERROR {e}")
                    sup_results[key] = float("nan")

        df_sup = pd.Series(sup_results, name="AUROC").to_frame()
        df_sup.to_csv(os.path.join(sup_dir, f"{category}_supervised.csv"))
        print(df_sup.to_string())

    elapsed = time.time() - t0
    print(f"\n[{category}] Done in {elapsed:.1f}s")
    return df_unsup


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Texture-based anomaly detection pipeline — MVTec"
    )
    p.add_argument("--dataset", required=True,
                   help="Path to MVTec root directory")
    p.add_argument("--category", default=None,
                   help="Single category name (e.g. wood, tile, grid)")
    p.add_argument("--all-categories", action="store_true",
                   help="Run on all categories found in --dataset")
    p.add_argument("--features", nargs="+", default=None,
                   help="Feature groups to use (default: all). "
                        "Options: " + ", ".join(FEATURE_GROUPS.keys()))
    p.add_argument("--detectors", nargs="+", default=None,
                   help="Detectors to use (default: all unsupervised). "
                        "Options: " + ", ".join(UNSUPERVISED_DETECTORS.keys()))
    p.add_argument("--out", default="results",
                   help="Output directory for results (default: results/)")
    p.add_argument("--img-size", type=int, default=256,
                   help="Resize images to this square size (default: 256)")
    p.add_argument("--denoise", default="gaussian",
                   choices=["gaussian", "median", "bilateral", "nlmeans",
                            "wavelet", "rclbp", "none"],
                   help="Denoising method (rclbp = NLMeans + wavelet, "
                        "from Gyimah et al. 2021)")
    p.add_argument("--enhance", default="clahe",
                   choices=["clahe", "histeq", "none"],
                   help="Contrast enhancement method")
    return p.parse_args()


def main():
    args = parse_args()

    # Determine categories
    if args.all_categories:
        categories = list_categories(args.dataset)
    elif args.category:
        categories = [args.category]
    else:
        print("Specify --category <name> or --all-categories")
        return

    # Determine feature groups
    if args.features:
        feat_groups = {k: FEATURE_GROUPS[k] for k in args.features
                       if k in FEATURE_GROUPS}
    else:
        feat_groups = FEATURE_GROUPS

    img_size = (args.img_size, args.img_size)
    denoise  = None if args.denoise == "none" else args.denoise
    enhance  = None if args.enhance == "none" else args.enhance

    all_results = {}
    for cat in categories:
        df = run_category(
            dataset_root=args.dataset,
            category=cat,
            feature_groups=feat_groups,
            detector_names=args.detectors or [],
            out_root=args.out,
            img_size=img_size,
            denoise=denoise,
            enhance=enhance,
        )
        all_results[cat] = df

    # Summary across categories (mean AUROC per feature × detector)
    if len(all_results) > 1:
        stacked = pd.concat(all_results.values())
        mean_df = stacked.groupby(stacked.index).mean()
        mean_df.index.name = "Features"
        mean_path = os.path.join(args.out, "summary_mean_auroc.csv")
        mean_df.to_csv(mean_path)
        print(f"\nMean AUROC across categories saved to {mean_path}")
        print(mean_df.to_string())


if __name__ == "__main__":
    main()
