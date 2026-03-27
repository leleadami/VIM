"""
dataset.py
----------
MVTec Anomaly Detection dataset loader.

Expected directory layout:
    <root>/
        <category>/          e.g. wood, tile, carpet
            train/
                good/        defect-free training images
            test/
                good/        normal test images  (label 0)
                <defect>/    defective test images (label 1)
"""

import os
import cv2
import numpy as np
from pathlib import Path


def load_mvtec_category(root: str, category: str, img_size: tuple = (256, 256)):
    """
    Load train and test splits for a single MVTec category.

    Training set contains only normal images (one-class paradigm).
    Test set contains both normal and defective images.

    Parameters
    ----------
    root     : path to the dataset root directory
    category : category name (e.g. 'wood', 'tile', 'hazelnut')
    img_size : (height, width) target size for resizing

    Returns
    -------
    X_train : (N_train, H, W) uint8 grayscale — normal training images
    y_train : (N_train,)      all zeros by definition
    X_test  : (N_test,  H, W) uint8 grayscale — normal + defective
    y_test  : (N_test,)       0 = normal, 1 = defective
    meta    : list of dicts with 'defect' type and 'file' path
    """
    base = Path(root) / category

    # training: normal images only
    train_dir = base / "train" / "good"
    X_train, y_train = _load_split(train_dir, label=0, img_size=img_size)

    # test: all subfolders under test/
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
    """
    Load all .png images from a folder as grayscale uint8 arrays.

    Note: grayscale conversion discards colour information — colour-based
    defects (e.g. 'color' class on wood) may be less discriminable.
    """
    images, labels = [], []
    for path in sorted(folder.glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # cv2.resize expects (width, height) — note the axis order swap
        img = cv2.resize(img, (img_size[1], img_size[0]))
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)


def list_categories(root: str):
    """Return all category names found under root."""
    return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
