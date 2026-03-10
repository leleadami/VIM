"""
dataset.py
----------
MVTec Anomaly Detection dataset loader.

Expected directory structure:
    mvtec/
        <category>/
            train/
                good/
                    *.png
            test/
                good/
                    *.png
                <defect_type>/
                    *.png
"""

import os
import cv2
import numpy as np
from pathlib import Path


def load_mvtec_category(root: str, category: str, img_size: tuple = (256, 256)):
    """
    Load train and test splits for a single MVTec category.

    Parameters
    ----------
    root : str
        Path to the MVTec root directory.
    category : str
        Category name, e.g. 'hazelnut', 'tile', 'wood'.
    img_size : tuple
        Target (height, width) for resizing.

    Returns
    -------
    X_train : np.ndarray  shape (N_train, H, W)  — grayscale uint8
    y_train : np.ndarray  shape (N_train,)        — all zeros (normal)
    X_test  : np.ndarray  shape (N_test,  H, W)
    y_test  : np.ndarray  shape (N_test,)         — 0 normal / 1 defective
    meta    : list of dict  — filename and defect type per test sample
    """
    base = Path(root) / category

    # ── training (only "good" images) ──────────────────────────────────────
    train_dir = base / "train" / "good"
    X_train, y_train = _load_split(train_dir, label=0, img_size=img_size)

    # ── test ────────────────────────────────────────────────────────────────
    test_dir = base / "test"
    X_test_list, y_test_list, meta = [], [], []

    for defect_folder in sorted(test_dir.iterdir()):
        label = 0 if defect_folder.name == "good" else 1
        imgs, _ = _load_split(defect_folder, label=label, img_size=img_size)
        for i, img in enumerate(imgs):
            X_test_list.append(img)
            y_test_list.append(label)
            meta.append({"defect": defect_folder.name,
                         "file": str(sorted(defect_folder.glob("*.png"))[i])})

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    print(f"[{category}] train: {len(X_train)} | "
          f"test normal: {(y_test == 0).sum()} | "
          f"test defective: {(y_test == 1).sum()}")

    return X_train, y_train, X_test, y_test, meta


def _load_split(folder: Path, label: int, img_size: tuple):
    images, labels = [], []
    for path in sorted(folder.glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_size[1], img_size[0]))
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)


def list_categories(root: str):
    """Return all category names present in the MVTec root directory."""
    return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
