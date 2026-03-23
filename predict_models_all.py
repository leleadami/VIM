#!/usr/bin/env python3
"""
predict_models_all.py
---------------------
Valutazione batch di tutti i modelli addestrati (.pkl) su tutte le immagini
di test delle rispettive categorie.

Per ogni categoria stampa:
  - TP, FP, TN, FN
  - Accuracy, Precision, Recall, F1
  - Dettaglio per tipo di difetto

Uso:
    python predict_models_all.py
    python predict_models_all.py --category wood
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from preprocessing import preprocess
from feature_extraction import extract_features

_PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = _PROJECT_ROOT / "dataset"
MODELS_DIR   = _PROJECT_ROOT / "models"
IMG_SIZE     = (256, 256)
DENOISE      = "gaussian"
ENHANCE      = "clahe"


def evaluate_category(category: str) -> dict:
    """Valuta il modello .pkl su tutte le immagini di test della categoria."""
    model_path = MODELS_DIR / f"{category}_best.pkl"
    if not model_path.exists():
        print(f"  [skip] Modello non trovato: {model_path}")
        return None

    bundle = joblib.load(model_path)
    detector = bundle["detector"]
    features = bundle["features"]
    threshold = float(detector._threshold)

    test_dir = DATASET_ROOT / category / "test"
    if not test_dir.exists():
        print(f"  [skip] Cartella test non trovata: {test_dir}")
        return None

    # Raccogli tutte le immagini di test
    tp = fp = tn = fn = 0
    per_defect = {}   # tipo_difetto -> {tp, fp, tn, fn, total}

    defect_folders = sorted(test_dir.iterdir())
    for folder in defect_folders:
        if not folder.is_dir():
            continue

        is_normal = (folder.name == "good")
        true_label = 0 if is_normal else 1
        defect_name = folder.name

        if defect_name not in per_defect:
            per_defect[defect_name] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0}

        images = sorted(folder.glob("*.png"))
        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img_pp = preprocess(img, denoise=DENOISE, enhance=ENHANCE)
            feat = extract_features(img_pp, features).reshape(1, -1)

            raw_score = float(detector.score_samples(feat)[0])
            pred_defect = raw_score >= threshold  # True = difettoso

            per_defect[defect_name]["total"] += 1

            if true_label == 1 and pred_defect:
                tp += 1
                per_defect[defect_name]["tp"] += 1
            elif true_label == 0 and pred_defect:
                fp += 1
                per_defect[defect_name]["fp"] += 1
            elif true_label == 0 and not pred_defect:
                tn += 1
                per_defect[defect_name]["tn"] += 1
            else:  # true_label == 1 and not pred_defect
                fn += 1
                per_defect[defect_name]["fn"] += 1

    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "category": category,
        "features": "+".join(f.upper() for f in features),
        "detector": bundle.get("detector_name", "?"),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "per_defect": per_defect,
    }


def print_results(res: dict):
    """Stampa i risultati per una categoria."""
    cat = res["category"].upper()
    w = 55

    print(f"\n{'=' * w}")
    print(f"  {cat}  —  {res['features']} + {res['detector']}")
    print(f"{'=' * w}")

    # Confusion matrix
    print(f"\n  {'Confusion Matrix':^40}")
    print(f"  {'':>20} {'Pred NORM':>12} {'Pred DIFET':>12}")
    print(f"  {'Vero NORM':>20} {res['tn']:>12d} {res['fp']:>12d}")
    print(f"  {'Vero DIFET':>20} {res['fn']:>12d} {res['tp']:>12d}")

    # Metriche
    print(f"\n  Totale immagini : {res['total']}")
    print(f"  Accuracy        : {res['accuracy']:.4f}")
    print(f"  Precision       : {res['precision']:.4f}")
    print(f"  Recall          : {res['recall']:.4f}")
    print(f"  F1-score        : {res['f1']:.4f}")

    # Dettaglio per tipo di difetto
    print(f"\n  {'Tipo difetto':<18} {'Tot':>5} {'Corr':>6} {'Err':>6} {'Rate':>9}")
    print(f"  {'-'*48}")
    for defect, d in sorted(res["per_defect"].items()):
        tot = d["total"]
        if defect == "good":
            # Normali: corretti = TN (predetti normali), errori = FP (predetti difettosi)
            corr = d["tn"]
            err  = d["fp"]
            rate_val = corr / tot if tot else 0
            label = "TN rate"
        else:
            # Difettosi: corretti = TP (predetti difettosi), errori = FN (predetti normali)
            corr = d["tp"]
            err  = d["fn"]
            rate_val = corr / tot if tot else 0
            label = "detect"
        print(f"  {defect:<18} {tot:>5} {corr:>6} {err:>6} {rate_val:>7.1%} {label}")


def main():
    p = argparse.ArgumentParser(description="Valutazione batch dei modelli su test set.")
    p.add_argument("--category", default=None,
                   choices=["wood", "tile", "grid", "hazelnut", "carpet", "leather"],
                   help="Valuta solo questa categoria (default: tutte)")
    args = p.parse_args()

    categories = [args.category] if args.category else \
        ["wood", "tile", "grid", "hazelnut", "carpet", "leather"]

    all_results = []

    for cat in categories:
        cat_dir = DATASET_ROOT / cat
        if not cat_dir.exists():
            print(f"[skip] {cat} — cartella dataset non trovata")
            continue

        print(f"\nElaborazione {cat}...", flush=True)
        res = evaluate_category(cat)
        if res is None:
            continue
        all_results.append(res)
        print_results(res)

    # Riepilogo finale
    if len(all_results) > 1:
        print(f"\n\n{'#' * 55}")
        print(f"  RIEPILOGO GLOBALE")
        print(f"{'#' * 55}")
        print(f"\n  {'Categoria':<12} {'Modello':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print(f"  {'-'*70}")
        for r in all_results:
            model = f"{r['features']}+{r['detector']}"
            if len(model) > 26:
                model = model[:24] + ".."
            print(f"  {r['category']:<12} {model:<28} "
                  f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} "
                  f"{r['recall']:>7.4f} {r['f1']:>7.4f}")

        # Totali aggregati
        tot_tp = sum(r["tp"] for r in all_results)
        tot_fp = sum(r["fp"] for r in all_results)
        tot_tn = sum(r["tn"] for r in all_results)
        tot_fn = sum(r["fn"] for r in all_results)
        tot = tot_tp + tot_fp + tot_tn + tot_fn
        acc = (tot_tp + tot_tn) / tot if tot else 0
        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0
        rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"  {'-'*70}")
        print(f"  {'TOTALE':<12} {'':28} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")
        print(f"\n  TP={tot_tp}  FP={tot_fp}  TN={tot_tn}  FN={tot_fn}  ({tot} immagini)")


if __name__ == "__main__":
    main()
