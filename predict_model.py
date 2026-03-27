#!/usr/bin/env python3
"""
predict_model.py
----------------
Predict whether a surface image is normal or defective.

On first run the model is trained from the dataset and cached in models/.
Subsequent runs load the cached model in under a second.

Usage
-----
    python predict_model.py image.png
    python predict_model.py image.png --category tile
    python predict_model.py image.png --category wood --retrain
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import joblib
import matplotlib
matplotlib.use("TkAgg")          # interactive desktop window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# project modules
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from preprocessing      import preprocess
from feature_extraction import extract_features
from dataset            import load_mvtec_category
from models             import UNSUPERVISED_DETECTORS

# constants
_PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = _PROJECT_ROOT / "dataset"
MODELS_DIR   = _PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DENOISE     = "gaussian"
ENHANCE     = "clahe"
IMG_SIZE    = (256, 256)

# map CSV feature-group names to feature_extraction keys
_FEAT_GROUP_MAP = {
    "Gabor": ["gabor"], "GLCM": ["glcm"], "HOG": ["hog"],
    "FFT": ["fft"], "Stats": ["stats"], "Laws": ["laws"],
    "Stats+GLCM+Gabor": ["stats", "glcm", "gabor"],
    "Stats+Laws+GLCM": ["stats", "laws", "glcm"],
    "All": ["gabor", "glcm", "hog", "fft", "stats", "laws"],
}


def _find_best_combo(category: str) -> tuple:
    """
    Read the AUROC CSV produced by pipeline.py and return the best
    (feature_list, detector_name, auroc) combination for the category.
    Exits with an error if no CSV is found.
    """
    unsup_dir = _PROJECT_ROOT / "results" / category / "unsupervised"

    cat_dir = _PROJECT_ROOT / "results" / category

    # collect all AUROC CSVs across preprocessing sub-directories
    all_csvs = list(cat_dir.glob(f"**/{category}_*auroc.csv"))

    if not all_csvs:
        sys.exit(f"Error: no AUROC CSV found for '{category}'.\n"
                 f"  Directory: {cat_dir}\n"
                 f"  Run first: python pipeline.py --dataset dataset --category {category}")

    # pick the CSV with the highest peak AUROC
    import pandas as pd
    best_csv = None
    best_global_auroc = -1.0
    for c in all_csvs:
        try:
            df_tmp = pd.read_csv(c, index_col=0)
            valid_cols = [col for col in df_tmp.columns if col in UNSUPERVISED_DETECTORS
                          and col != "PCANullSubspace"]
            if not valid_cols:
                continue
            max_val = df_tmp[valid_cols].max().max()
            if max_val > best_global_auroc:
                best_global_auroc = max_val
                best_csv = c
        except Exception:
            continue

    csv_path = best_csv
    print(f"[auto] Best CSV found: {csv_path.name} (peak AUROC={best_global_auroc:.4f})")

    import pandas as pd
    df = pd.read_csv(csv_path, index_col=0)

    # find the cell with the highest AUROC (PCANullSubspace excluded — needs fit params)
    valid_detectors = [c for c in df.columns if c in UNSUPERVISED_DETECTORS
                       and c != "PCANullSubspace"]
    df_valid = df[valid_detectors]

    best_auroc = -1.0
    best_feat = None
    best_det = None
    for feat_name in df_valid.index:
        for det_name in df_valid.columns:
            val = df_valid.loc[feat_name, det_name]
            if pd.notna(val) and val > best_auroc:
                best_auroc = val
                best_feat = feat_name
                best_det = det_name

    feat_list = _FEAT_GROUP_MAP.get(best_feat, ["stats"])
    print(f"[auto] Migliore dal CSV: {best_feat} + {best_det} = AUROC {best_auroc:.4f}")
    return feat_list, best_det, best_auroc


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_save(category: str, features: list = None,
                   detector_name: str = None) -> dict:
    """
    Train the best detector for a category and cache it to disk.
    Uses the same SklearnDetector pipeline as pipeline.py.
    Returns the serialised bundle {detector, features, ...}.
    """
    # auto-select best combo from CSV when not explicitly specified
    if features is None or detector_name is None:
        auto_feat, auto_det, auto_auroc = _find_best_combo(category)
        features = features or auto_feat
        detector_name = detector_name or auto_det

    print(f"\n[train] Carico immagini normali di '{category}'...")
    X_raw, y_train, _, _, _ = load_mvtec_category(
        str(DATASET_ROOT), category, img_size=IMG_SIZE
    )

    X_normal = X_raw[y_train == 0]
    print(f"[train] {len(X_normal)} normal images found")

    print("[train] Preprocessing...")
    from preprocessing import preprocess_batch
    X_pp = preprocess_batch(X_normal, denoise=DENOISE, enhance=ENHANCE)

    feat_label = "+".join(f.upper() for f in features)
    print(f"[train] Extracting features ({feat_label})...")
    X_feat = np.vstack([extract_features(img, features) for img in X_pp])
    print(f"[train] Feature shape: {X_feat.shape}")

    detector_factory = UNSUPERVISED_DETECTORS[detector_name]
    detector = detector_factory()
    print(f"[train] Detector: {detector_name}")
    detector.fit(X_feat)

    # score range on training normals — used for [0,1] normalisation at predict time
    train_scores = detector.score_samples(X_feat)
    score_min    = float(train_scores.min())
    score_max    = float(train_scores.max())

    bundle = dict(detector=detector,
                  score_min=score_min, score_max=score_max,
                  features=features, detector_name=detector_name,
                  category=category)

    path = MODELS_DIR / f"{category}_best.pkl"
    joblib.dump(bundle, path)
    print(f"[train] Model saved -> {path}\n")
    return bundle


def load_model(category: str) -> dict:
    """Load cached model from disk, training it first if not found."""
    path = MODELS_DIR / f"{category}_best.pkl"
    if not path.exists():
        return train_and_save(category)
    print(f"[model] Loading model from {path}")
    return joblib.load(path)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(image_path: str, bundle: dict) -> dict:
    """
    Preprocess an image, extract features, and return an anomaly verdict.

    Returns a dict with:
        score      : normalised anomaly score in [0, 1]  (1 = certain anomaly)
        raw_score  : raw detector score
        label      : "NORMALE" or "DIFETTOSO"
        confidence : confidence percentage
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        sys.exit(f"Error: cannot read '{image_path}'")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, IMG_SIZE)
    img_pp   = preprocess(img_gray, denoise=DENOISE, enhance=ENHANCE)

    feat = extract_features(img_pp, bundle["features"]).reshape(1, -1)

    detector = bundle["detector"]
    raw_score = float(detector.score_samples(feat)[0])
    thr       = float(detector._threshold)
    s_min     = bundle["score_min"]
    s_max     = bundle["score_max"]

    # Classification: score >= threshold → anomalous
    is_defect = raw_score >= thr
    label     = "DIFETTOSO" if is_defect else "NORMALE"

    # Normalisation: threshold maps to 0.5, scale based on training range
    span = max(s_max - s_min, 1e-9)
    norm_score = 0.5 + (raw_score - thr) / (2.0 * span)
    norm_score = float(np.clip(norm_score, 0.0, 1.0))

    thr_norm = 0.5  # threshold always maps to 0.5 in normalised space

    # confidence: distance from threshold scaled to [50%, 99%]
    dist = abs(raw_score - thr) / (span + 1e-9)
    confidence = float(np.clip(50 + dist * 100, 50, 99))

    return dict(
        image_path  = image_path,
        img_display = img_gray,
        img_pp      = img_pp,
        raw_score   = raw_score,
        norm_score  = norm_score,
        threshold   = thr,
        thr_norm    = thr_norm,
        label       = label,
        is_defect   = is_defect,
        confidence  = confidence,
        category    = bundle["category"],
        detector_name = bundle.get("detector_name", "IsolationForest"),
        features_label = "+".join(f.upper() for f in bundle.get("features", ["stats"])),
    )


# ── Visualisation ────────────────────────────────────────────────────────────

def show_result(res: dict):
    """Print the result to terminal and open a matplotlib gauge window."""

    color  = "\033[91m" if res["is_defect"] else "\033[92m"
    reset  = "\033[0m"
    symbol = "[X]" if res["is_defect"] else "[OK]"

    print("\n" + "─"*50)
    print(f"  File     : {Path(res['image_path']).name}")
    print(f"  Categoria: {res['category']}")
    print(f"  Modello  : {res['features_label']} + {res['detector_name']}")
    print(f"  Score    : {res['norm_score']:.3f}  "
          f"(soglia={res['thr_norm']:.3f})")
    print(f"  Risultato: {color}{symbol}  {res['label']}{reset}  "
          f"({res['confidence']:.0f}% confidenza)")
    print("─"*50 + "\n")

    # ── finestra matplotlib ───────────────────────────────────────────────────
    BG       = "#1e1e2e"
    bar_color = "#ff4444" if res["is_defect"] else "#44cc88"
    border_color = "#ff4444" if res["is_defect"] else "#44ff88"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             gridspec_kw={"width_ratios": [1, 1.1]})
    fig.patch.set_facecolor(BG)

    # ── pannello sinistro: immagine preprocessata ─────────────────────────────
    ax_img = axes[0]
    ax_img.imshow(res["img_pp"], cmap="gray", vmin=0, vmax=255)
    ax_img.set_title("Immagine preprocessata", color="white",
                     fontsize=11, pad=8)
    ax_img.axis("off")
    ax_img.set_facecolor(BG)

    # coloured border around the image panel
    rect = mpatches.FancyBboxPatch(
        (-0.5, -0.5),
        res["img_pp"].shape[1] - 0.5, res["img_pp"].shape[0] - 0.5,
        boxstyle="square,pad=0", linewidth=5,
        edgecolor=border_color, facecolor="none",
        transform=ax_img.transData, clip_on=False
    )
    ax_img.add_patch(rect)

    # right panel: anomaly gauge
    ax = axes[1]
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    ax.axis("off")

    thr  = res["thr_norm"]
    sc   = res["norm_score"]
    emoji = "DIFETTOSO" if res["is_defect"] else "NORMALE"

    # verdict label
    ax.text(0.5, 9.2, emoji,
            ha="center", va="center",
            fontsize=24, fontweight="bold", color=bar_color)

    # score and confidence text
    ax.text(0.5, 7.9,
            f"Anomaly score:  {sc:.3f}",
            ha="center", va="center",
            fontsize=13, color="white")

    ax.text(0.5, 7.0,
            f"Confidenza:  {res['confidence']:.0f}%",
            ha="center", va="center",
            fontsize=11, color="#aaaaaa")

    # gauge bar
    BAR_Y = 4.8
    BAR_H = 1.2

    ax.barh(BAR_Y, 1.0, height=BAR_H, color="#2a2a44", left=0, zorder=2)
    ax.barh(BAR_Y, thr, height=BAR_H, color="#1a3a2a", left=0, zorder=3)
    ax.barh(BAR_Y, 1.0 - thr, height=BAR_H, color="#3a1a1a", left=thr, zorder=3)
    ax.barh(BAR_Y, sc, height=BAR_H * 0.55,
            color=bar_color, left=0, zorder=4, alpha=0.95)

    # threshold marker
    half_h = BAR_H * 0.65
    ax.plot([thr, thr], [BAR_Y - half_h, BAR_Y + half_h],
            color="#ffdd00", linestyle="--", linewidth=2.5, zorder=5)

    # scale labels: 0 / threshold / 1
    label_y = BAR_Y - BAR_H * 0.85
    ax.text(0.0,  label_y, "0",          ha="center", va="top",
            fontsize=9, color="#777799")
    ax.text(thr,  label_y, f"⚑ {thr:.2f}", ha="center", va="top",
            fontsize=9, color="#ffdd00")
    ax.text(1.0,  label_y, "1",          ha="center", va="top",
            fontsize=9, color="#777799")

    # score position indicator
    ax.plot(sc, BAR_Y + BAR_H * 0.58, marker="v",
            color="white", markersize=9, zorder=6)

    # horizontal separator between text and gauge
    ax.axhline(6.2, color="#333355", linewidth=1, xmin=0.02, xmax=0.98)

    fig.suptitle(
        f"Anomaly Detection — {res['category'].upper()}  |  "
        f"{Path(res['image_path']).name}",
        color="#ccccdd", fontsize=11, y=1.00
    )
    plt.tight_layout(pad=1.5)

    # save figure — name: category_defecttype_stem.png
    img_path = Path(res['image_path']).resolve()
    category = res['category']
    # extract defect type from path (e.g. dataset/tile/test/rough/001.png)
    defect_type = "unknown"
    parts = img_path.parts
    for i, part in enumerate(parts):
        if part == "test" and i + 1 < len(parts):
            defect_type = parts[i + 1]
            break
    out_name = f"{category}_{defect_type}_{img_path.stem}.png"
    out_path = _PROJECT_ROOT / "results" / out_name
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Result saved → {out_path}")

    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _guess_category(image_path: str) -> str:
    """
    Infer the MVTec category from the image path.
    E.g. 'dataset/tile/test/rough/001.png' → 'tile'
    """
    known = {"wood", "tile", "grid", "hazelnut", "carpet", "leather"}
    parts = Path(image_path).resolve().parts
    for part in parts:
        if part in known:
            return part
    return None


def train_all_categories():
    """Train and save the best model for every category."""
    categories = ["wood", "tile", "grid", "hazelnut", "carpet", "leather"]
    for cat in categories:
        cat_dir = DATASET_ROOT / cat
        if not cat_dir.exists():
            print(f"[skip] {cat} — directory not found")
            continue
        print(f"\n{'='*50}")
        print(f"  {cat.upper()}")
        print(f"{'='*50}")
        train_and_save(cat)
    print("\n[done] All models saved to", MODELS_DIR)


def main():
    p = argparse.ArgumentParser(
        description="Predict whether a surface image is normal or defective."
    )
    p.add_argument("image", nargs="?", default=None,
                   help="Path to the image to analyse")
    p.add_argument("--category", default=None,
                   choices=["wood", "tile", "grid", "hazelnut", "carpet", "leather"],
                   help="MVTec category (inferred from path if omitted)")
    p.add_argument("--retrain", action="store_true",
                   help="Force retraining even if a saved model exists")
    p.add_argument("--train-all", action="store_true",
                   help="Train and save the best model for every category")
    args = p.parse_args()

    if args.train_all:
        train_all_categories()
        return

    if args.image is None:
        p.error("image path required (or use --train-all)")

    if not Path(args.image).exists():
        sys.exit(f"Error: file '{args.image}' not found.")

    # infer category from path if not specified
    category = args.category or _guess_category(args.image)
    if category is None:
        sys.exit("Error: cannot infer category from path. Use --category <name>.")
    print(f"[info] Category: {category}")

    # load (or train) the model
    model_path = MODELS_DIR / f"{category}_best.pkl"
    if args.retrain and model_path.exists():
        model_path.unlink()

    bundle = load_model(category)

    res = predict(args.image, bundle)

    show_result(res)


if __name__ == "__main__":
    main()
