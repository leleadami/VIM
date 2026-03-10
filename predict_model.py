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
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing      import preprocess
from feature_extraction import extract_features
from dataset            import load_mvtec_category

# ── costanti ──────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parent / "dataset"
MODELS_DIR   = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES    = ["stats"]          # migliori per wood secondo gli esperimenti
DENOISE     = "gaussian"
ENHANCE     = "clahe"
IMG_SIZE    = (256, 256)


# ═════════════════════════════════════════════════════════════════════════════
# Training & salvataggio modello
# ═════════════════════════════════════════════════════════════════════════════

def train_and_save(category: str) -> dict:
    """
    Addestra il modello sulla categoria indicata e lo salva su disco.
    Restituisce il bundle {scaler, pca, detector, threshold}.
    """
    from sklearn.preprocessing        import StandardScaler
    from sklearn.decomposition        import PCA
    from sklearn.ensemble             import IsolationForest

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
    print("[train] Estrazione feature (Stats)...")
    X_feat = np.vstack([extract_features(img, FEATURES) for img in X_pp])
    print(f"[train] Feature shape: {X_feat.shape}")

    # standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # PCA
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"[train] PCA: {X_feat.shape[1]} → {X_pca.shape[1]} componenti "
          f"(95% varianza)")

    # Isolation Forest
    detector = IsolationForest(n_estimators=200, contamination=0.05,
                               random_state=42)
    detector.fit(X_pca)

    # soglia: 5° percentile degli score sui normali
    # (score_samples restituisce valori negativi: più negativo = più anomalo)
    train_scores = detector.score_samples(X_pca)
    threshold    = float(np.percentile(train_scores, 5))
    score_min    = float(train_scores.min())
    score_max    = float(train_scores.max())

    bundle = dict(scaler=scaler, pca=pca, detector=detector,
                  threshold=threshold,
                  score_min=score_min, score_max=score_max,
                  features=FEATURES, category=category)

    path = MODELS_DIR / f"{category}_stats_iforest.pkl"
    joblib.dump(bundle, path)
    print(f"[train] Modello salvato → {path}\n")
    return bundle


def load_model(category: str) -> dict:
    path = MODELS_DIR / f"{category}_stats_iforest.pkl"
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

    # pipeline: scala → PCA → score
    feat_scaled = bundle["scaler"].transform(feat)
    feat_pca    = bundle["pca"].transform(feat_scaled)
    raw_score   = float(bundle["detector"].score_samples(feat_pca)[0])

    # normalizza in [0, 1]:  0 = sicuramente normale, 1 = sicuramente anomalo
    # IsolationForest: score_max ≈ 0 (normale), score_min < 0 (anomalo)
    s_min = bundle["score_min"]
    s_max = bundle["score_max"]
    norm_score = 1.0 - (raw_score - s_min) / (s_max - s_min + 1e-9)
    norm_score = float(np.clip(norm_score, 0.0, 1.0))

    # soglia normalizzata
    thr_norm = 1.0 - (bundle["threshold"] - s_min) / (s_max - s_min + 1e-9)
    thr_norm = float(np.clip(thr_norm, 0.0, 1.0))

    is_defect  = raw_score < bundle["threshold"]
    label      = "DIFETTOSO" if is_defect else "NORMALE"

    # distanza dalla soglia → confidenza
    dist = abs(raw_score - bundle["threshold"]) / (abs(s_min - s_max) + 1e-9)
    confidence = float(np.clip(50 + dist * 100, 50, 99))

    return dict(
        image_path  = image_path,
        img_display = img_gray,
        img_pp      = img_pp,
        raw_score   = raw_score,
        norm_score  = norm_score,
        threshold   = bundle["threshold"],
        thr_norm    = thr_norm,
        label       = label,
        is_defect   = is_defect,
        confidence  = confidence,
        category    = bundle["category"],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Output visivo
# ═════════════════════════════════════════════════════════════════════════════

def show_result(res: dict):
    """Stampa il risultato a terminale e apre una finestra grafica."""

    # ── stampa terminale ──────────────────────────────────────────────────────
    color  = "\033[91m" if res["is_defect"] else "\033[92m"
    reset  = "\033[0m"
    symbol = "🔴" if res["is_defect"] else "🟢"

    print("\n" + "─"*46)
    print(f"  File     : {Path(res['image_path']).name}")
    print(f"  Categoria: {res['category']}")
    print(f"  Score    : {res['norm_score']:.3f}  "
          f"(soglia={res['thr_norm']:.3f})")
    print(f"  Risultato: {color}{symbol}  {res['label']}{reset}  "
          f"({res['confidence']:.0f}% confidenza)")
    print("─"*46 + "\n")

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
    emoji = "🔴  DIFETTOSO" if res["is_defect"] else "🟢  NORMALE"

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

    # salva come file
    out_path = Path("results") / f"prediction_{Path(res['image_path']).stem}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Risultato salvato → {out_path}")

    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Predice se una superficie è normale o difettosa."
    )
    p.add_argument("image", help="Percorso dell'immagine da analizzare")
    p.add_argument("--category", default="wood",
                   choices=["wood", "tile", "grid", "hazelnut", "carpet", "leather"],
                   help="Categoria MVTec su cui è stato addestrato il modello "
                        "(default: wood)")
    p.add_argument("--retrain", action="store_true",
                   help="Forza il ri-addestramento anche se il modello esiste")
    args = p.parse_args()

    if not Path(args.image).exists():
        sys.exit(f"Errore: file '{args.image}' non trovato.")

    # carica (o addestra) il modello
    model_path = MODELS_DIR / f"{args.category}_stats_iforest.pkl"
    if args.retrain and model_path.exists():
        model_path.unlink()

    bundle = load_model(args.category)

    # predizione
    res = predict(args.image, bundle)

    # output
    show_result(res)


if __name__ == "__main__":
    main()
