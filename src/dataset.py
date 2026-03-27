"""
dataset.py
----------
Caricatore del dataset MVTec Anomaly Detection.

Struttura attesa sul disco:
    dataset/
        <categoria>/          es. wood, tile, carpet
            train/
                good/         solo immagini normali per l'addestramento
                    *.png
            test/
                good/         normali di test (label 0)
                    *.png
                <tipo_difetto>/   es. scratch, hole, color (label 1)
                    *.png
"""

import os
import cv2
import numpy as np
from pathlib import Path


def load_mvtec_category(root: str, category: str, img_size: tuple = (256, 256)):
    """
    Carica train e test per una singola categoria MVTec.

    Il training set contiene SOLO immagini normali (folder 'good').
    Il test set contiene normali + difettose (tutte le sottocartelle di test/).

    Parametri
    ----------
    root     : percorso alla cartella radice del dataset (es. 'dataset/')
    category : nome categoria (es. 'wood', 'tile', 'hazelnut')
    img_size : dimensione target (altezza, larghezza) per il resize

    Ritorna
    -------
    X_train : array (N_train, H, W)  — immagini normali di training, uint8 grayscale
    y_train : array (N_train,)       — tutti 0 (normale per definizione)
    X_test  : array (N_test,  H, W)  — immagini di test (normali + difettose)
    y_test  : array (N_test,)        — 0 = normale, 1 = difettosa
    meta    : lista di dict con 'defect' (tipo difetto) e 'file' (percorso)
    """
    base = Path(root) / category

    # ── carica le immagini di training (solo normali) ─────────────────────────
    train_dir = base / "train" / "good"
    X_train, y_train = _load_split(train_dir, label=0, img_size=img_size)

    # ── carica le immagini di test (normali + difettose) ──────────────────────
    test_dir = base / "test"
    X_test_list, y_test_list, meta = [], [], []

    for defect_folder in sorted(test_dir.iterdir()):
        # la cartella 'good' contiene i normali di test (label 0)
        # tutte le altre cartelle sono tipi di difetto (label 1)
        label = 0 if defect_folder.name == "good" else 1
        imgs, _ = _load_split(defect_folder, label=label, img_size=img_size)
        for i, img in enumerate(imgs):
            X_test_list.append(img)
            y_test_list.append(label)
            # meta conserva il nome del difetto — utile per analisi per-tipo
            meta.append({"defect": defect_folder.name,
                         "file": str(sorted(defect_folder.glob("*.png"))[i])})

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    print(f"[{category}] train: {len(X_train)} | "
          f"test normal: {(y_test == 0).sum()} | "
          f"test defective: {(y_test == 1).sum()}")

    # il modello impara la distribuzione normale da X_train,
    # poi assegna uno score anomalia a ogni immagine di X_test.
    # y_test serve solo per valutare quanto è bravo (AUROC, F1...).
    return X_train, y_train, X_test, y_test, meta


def _load_split(folder: Path, label: int, img_size: tuple):
    """
    Carica tutte le immagini .png da una cartella.

    Le immagini vengono convertite in grayscale e ridimensionate.
    Nota: la conversione grayscale perde informazioni cromatiche —
    difetti come 'color' su wood (macchie di colore) diventano
    meno visibili se la luminosità è simile al fondo.
    """
    images, labels = [], []
    for path in sorted(folder.glob("*.png")):
        # IMREAD_GRAYSCALE converte direttamente in scala di grigi
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # OpenCV vuole (larghezza, altezza) — nota lo swap rispetto a NumPy (H,W)
        img = cv2.resize(img, (img_size[1], img_size[0]))
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)


def list_categories(root: str):
    """Restituisce tutti i nomi di categoria presenti nella cartella root."""
    return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
