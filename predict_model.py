#!/usr/bin/env python3
"""
predict_model.py
----------------
Dato un'immagine, dice se la superficie è NORMALE o DIFETTOSA.

Il modello migliore dai nostri esperimenti:
    Features : Statistical Moments  (gradiente Sobel: media, std, skew, kurt)
    Detector : Isolation Forest
    AUROC    : 0.9316 su wood

Al primo avvio addestra il modello dal dataset e lo salva in models/.
Dai avvii successivi carica il modello già addestrato (< 1 secondo).

Uso
---
    python predict_model.py immagine.png
    python predict_model.py immagine.png --category tile
    python predict_model.py immagine.png --category wood --retrain
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import joblib
import matplotlib
matplotlib.use("TkAgg")          # usa finestra desktop
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── moduli del progetto ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from preprocessing      import preprocess
from feature_extraction import extract_features
from dataset            import load_mvtec_category
from models             import UNSUPERVISED_DETECTORS

# ── costanti ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = _PROJECT_ROOT / "dataset"
MODELS_DIR   = _PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DENOISE     = "gaussian"
ENHANCE     = "clahe"
IMG_SIZE    = (256, 256)

# Mappa nomi CSV → nomi feature_extraction
_FEAT_GROUP_MAP = {
    "LBP": ["lbp"], "LBP_MS": ["lbp_ms"], "CLBP": ["clbp"],
    "Gabor": ["gabor"], "GLCM": ["glcm"], "HOG": ["hog"],
    "FFT": ["fft"], "Wavelet": ["wavelet"], "Stats": ["stats"],
    "Laws": ["laws"], "DSIFT": ["dsift"],
    "LBP+GLCM+Gabor": ["lbp", "glcm", "gabor"],
    "LBP_MS+GLCM+Gabor": ["lbp_ms", "glcm", "gabor"],
    "Stats+GLCM+Gabor": ["stats", "glcm", "gabor"],
    "Stats+Laws+GLCM": ["stats", "laws", "glcm"],
    "All": ["lbp", "lbp_ms", "clbp", "gabor", "glcm", "hog", "fft", "wavelet", "stats", "laws", "dsift"],
}


def _find_best_combo(category: str) -> tuple:
    """
    Legge il CSV generato da pipeline.py e trova la combo
    feature+detector con AUROC massimo per la categoria.
    Restituisce (feature_list, detector_name, auroc).
    Se il CSV non esiste, esce con errore.
    """
    csv_path = _PROJECT_ROOT / "results" / category / "unsupervised" / f"{category}_unsupervised_auroc.csv"
    if not csv_path.exists():
        sys.exit(f"Errore: CSV non trovato per '{category}'.\n"
                 f"  Atteso: {csv_path}\n"
                 f"  Lancia prima: python pipeline.py --dataset dataset --category {category}")

    import pandas as pd
    df = pd.read_csv(csv_path, index_col=0)

    # Trova la cella con AUROC massimo (esclude PCANullSubspace)
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


# ═════════════════════════════════════════════════════════════════════════════
# Training & salvataggio modello
# ═════════════════════════════════════════════════════════════════════════════

def train_and_save(category: str, features: list = None,
                   detector_name: str = None) -> dict:
    """
    Addestra il modello sulla categoria indicata e lo salva su disco.
    Usa lo STESSO SklearnDetector di models.py (identico a pipeline).
    Restituisce il bundle {detector, features, ...}.
    """
    # Scegli combo migliore dal CSV se non specificata
    if features is None or detector_name is None:
        auto_feat, auto_det, auto_auroc = _find_best_combo(category)
        features = features or auto_feat
        detector_name = detector_name or auto_det

    print(f"\n[train] Carico immagini normali di '{category}'...")
    X_raw, y_train, _, _, _ = load_mvtec_category(
        str(DATASET_ROOT), category, img_size=IMG_SIZE
    )

    # solo immagini normali (y_train == 0)
    X_normal = X_raw[y_train == 0]
    print(f"[train] {len(X_normal)} immagini normali trovate")

    # preprocessing
    print("[train] Preprocessing...")
    from preprocessing import preprocess_batch
    X_pp = preprocess_batch(X_normal, denoise=DENOISE, enhance=ENHANCE)

    # feature extraction
    feat_label = "+".join(f.upper() for f in features)
    print(f"[train] Estrazione feature ({feat_label})...")
    X_feat = np.vstack([extract_features(img, features) for img in X_pp])
    print(f"[train] Feature shape: {X_feat.shape}")

    # Usa la STESSA factory di models.py (SklearnDetector con scaler+PCA interni)
    detector_factory = UNSUPERVISED_DETECTORS[detector_name]
    detector = detector_factory()
    print(f"[train] Detector: {detector_name}")
    detector.fit(X_feat)    # SklearnDetector fa scaler+PCA+fit internamente

    # Score sui normali di training (higher = more anomalous, gia' negato)
    train_scores = detector.score_samples(X_feat)
    score_min    = float(train_scores.min())
    score_max    = float(train_scores.max())

    bundle = dict(detector=detector,
                  score_min=score_min, score_max=score_max,
                  features=features, detector_name=detector_name,
                  category=category)

    path = MODELS_DIR / f"{category}_best.pkl"
    joblib.dump(bundle, path)
    print(f"[train] Modello salvato -> {path}\n")
    return bundle


def load_model(category: str) -> dict:
    path = MODELS_DIR / f"{category}_best.pkl"
    if not path.exists():
        return train_and_save(category)
    print(f"[model] Carico modello da {path}")
    return joblib.load(path)


# ═════════════════════════════════════════════════════════════════════════════
# Predizione su singola immagine
# ═════════════════════════════════════════════════════════════════════════════

def predict(image_path: str, bundle: dict) -> dict:
    """
    Preprocessa l'immagine, estrae le feature e restituisce:
        score      : anomaly score normalizzato in [0, 1]  (1 = anomalia certa)
        raw_score  : score grezzo di IsolationForest
        label      : "NORMALE" o "DIFETTOSO"
        confidence : percentuale di sicurezza
    """
    # carica e pre-processa
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        sys.exit(f"Errore: impossibile leggere '{image_path}'")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, IMG_SIZE)
    img_pp   = preprocess(img_gray, denoise=DENOISE, enhance=ENHANCE)

    # feature
    feat = extract_features(img_pp, bundle["features"]).reshape(1, -1)

    # Score via SklearnDetector (scaler+PCA+detector tutto interno, higher=more anomalous)
    detector = bundle["detector"]
    raw_score = float(detector.score_samples(feat)[0])
    thr       = float(detector._threshold)
    s_min     = bundle["score_min"]
    s_max     = bundle["score_max"]

    # Classificazione: score >= threshold → anomalo
    is_defect = raw_score >= thr
    label     = "DIFETTOSO" if is_defect else "NORMALE"

    # Normalizzazione: soglia = 0.5, scala basata sul range training
    # Cosi' anche score fuori dal range training vengono rappresentati
    span = max(s_max - s_min, 1e-9)
    norm_score = 0.5 + (raw_score - thr) / (2.0 * span)
    norm_score = float(np.clip(norm_score, 0.0, 1.0))

    # Soglia normalizzata = sempre 0.5
    thr_norm = 0.5

    # Confidenza: distanza dalla soglia
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


# ═════════════════════════════════════════════════════════════════════════════
# Output visivo
# ═════════════════════════════════════════════════════════════════════════════

def show_result(res: dict):
    """Stampa il risultato a terminale e apre una finestra grafica."""

    # ── stampa terminale ──────────────────────────────────────────────────────
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

    # bordo colorato: rettangolo attorno all'immagine
    rect = mpatches.FancyBboxPatch(
        (-0.5, -0.5),
        res["img_pp"].shape[1] - 0.5, res["img_pp"].shape[0] - 0.5,
        boxstyle="square,pad=0", linewidth=5,
        edgecolor=border_color, facecolor="none",
        transform=ax_img.transData, clip_on=False
    )
    ax_img.add_patch(rect)

    # ── pannello destro: gauge ────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    ax.axis("off")

    thr  = res["thr_norm"]
    sc   = res["norm_score"]
    emoji = "DIFETTOSO" if res["is_defect"] else "NORMALE"

    # -- etichetta risultato (in alto) ----------------------------------------
    ax.text(0.5, 9.2, emoji,
            ha="center", va="center",
            fontsize=24, fontweight="bold", color=bar_color)

    # -- score e confidenza (zona centrale) ------------------------------------
    ax.text(0.5, 7.9,
            f"Anomaly score:  {sc:.3f}",
            ha="center", va="center",
            fontsize=13, color="white")

    ax.text(0.5, 7.0,
            f"Confidenza:  {res['confidence']:.0f}%",
            ha="center", va="center",
            fontsize=11, color="#aaaaaa")

    # -- barra gauge (zona bassa) -----------------------------------------------
    BAR_Y = 4.8          # centro barra in coordinata dati
    BAR_H = 1.2          # altezza barra

    # sfondo barra (grigio scuro)
    ax.barh(BAR_Y, 1.0, height=BAR_H, color="#2a2a44", left=0, zorder=2)

    # zona "normale" (verde tenue) fino alla soglia
    ax.barh(BAR_Y, thr, height=BAR_H, color="#1a3a2a", left=0, zorder=3)

    # zona "difettoso" (rossa tenue) oltre la soglia
    ax.barh(BAR_Y, 1.0 - thr, height=BAR_H, color="#3a1a1a", left=thr, zorder=3)

    # barra score colorata (sopra lo sfondo)
    ax.barh(BAR_Y, sc, height=BAR_H * 0.55,
            color=bar_color, left=0, zorder=4, alpha=0.95)

    # linea soglia (gialla tratteggiata)
    half_h = BAR_H * 0.65
    ax.plot([thr, thr], [BAR_Y - half_h, BAR_Y + half_h],
            color="#ffdd00", linestyle="--", linewidth=2.5, zorder=5)

    # etichette scala: 0 / soglia / 1
    label_y = BAR_Y - BAR_H * 0.85
    ax.text(0.0,  label_y, "0",          ha="center", va="top",
            fontsize=9, color="#777799")
    ax.text(thr,  label_y, f"⚑ {thr:.2f}", ha="center", va="top",
            fontsize=9, color="#ffdd00")
    ax.text(1.0,  label_y, "1",          ha="center", va="top",
            fontsize=9, color="#777799")

    # indicatore triangolare sulla barra (posizione score)
    ax.plot(sc, BAR_Y + BAR_H * 0.58, marker="v",
            color="white", markersize=9, zorder=6)

    # separatore orizzontale tra testo e barra
    ax.axhline(6.2, color="#333355", linewidth=1, xmin=0.02, xmax=0.98)

    fig.suptitle(
        f"Anomaly Detection — {res['category'].upper()}  |  "
        f"{Path(res['image_path']).name}",
        color="#ccccdd", fontsize=11, y=1.00
    )
    plt.tight_layout(pad=1.5)

    # salva come file — nome: materiale_difetto_num.png
    img_path = Path(res['image_path']).resolve()
    category = res['category']
    # prova a estrarre il tipo difetto dal percorso (es. dataset/tile/test/rough/001.png)
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
    print(f"Risultato salvato → {out_path}")

    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _guess_category(image_path: str) -> str:
    """
    Deduce la categoria MVTec dal percorso dell'immagine.
    Es: 'dataset/tile/test/rough/001.png' → 'tile'
        '/home/.../dataset/wood/test/good/003.png' → 'wood'
    """
    known = {"wood", "tile", "grid", "hazelnut", "carpet", "leather"}
    parts = Path(image_path).resolve().parts
    for part in parts:
        if part in known:
            return part
    return None


def train_all_categories():
    """Addestra e salva il modello migliore per ogni categoria."""
    categories = ["wood", "tile", "grid", "hazelnut", "carpet", "leather"]
    for cat in categories:
        cat_dir = DATASET_ROOT / cat
        if not cat_dir.exists():
            print(f"[skip] {cat} — cartella non trovata")
            continue
        print(f"\n{'='*50}")
        print(f"  {cat.upper()}")
        print(f"{'='*50}")
        train_and_save(cat)
    print("\n[done] Tutti i modelli salvati in", MODELS_DIR)


def main():
    p = argparse.ArgumentParser(
        description="Predice se una superficie è normale o difettosa."
    )
    p.add_argument("image", nargs="?", default=None,
                   help="Percorso dell'immagine da analizzare")
    p.add_argument("--category", default=None,
                   choices=["wood", "tile", "grid", "hazelnut", "carpet", "leather"],
                   help="Categoria MVTec (se omesso, la deduce dal percorso)")
    p.add_argument("--retrain", action="store_true",
                   help="Forza il ri-addestramento anche se il modello esiste")
    p.add_argument("--train-all", action="store_true",
                   help="Addestra e salva il modello migliore per ogni categoria")
    args = p.parse_args()

    # Modalita' addestra tutto
    if args.train_all:
        train_all_categories()
        return

    if args.image is None:
        p.error("serve il percorso dell'immagine (oppure usa --train-all)")

    if not Path(args.image).exists():
        sys.exit(f"Errore: file '{args.image}' non trovato.")

    # deduce categoria dal path se non specificata
    category = args.category or _guess_category(args.image)
    if category is None:
        sys.exit("Errore: impossibile dedurre la categoria dal percorso. "
                 "Usa --category <nome>.")
    print(f"[info] Categoria: {category}")

    # carica (o addestra) il modello
    model_path = MODELS_DIR / f"{category}_best.pkl"
    if args.retrain and model_path.exists():
        model_path.unlink()

    bundle = load_model(category)

    # predizione
    res = predict(args.image, bundle)

    # output
    show_result(res)


if __name__ == "__main__":
    main()
