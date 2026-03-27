"""
feature_extraction.py
---------------------
Tutti gli estrattori di feature testurali. Ogni funzione prende una singola
immagine grayscale uint8 (H×W) e restituisce un vettore float64 1-D.

Estrattori disponibili
----------------------
- gabor_features      : Bank di filtri Gabor — analisi frequenza/orientazione
- glcm_features       : Gray-Level Co-occurrence Matrix — statistiche texture
- hog_features        : Histogram of Oriented Gradients — struttura dei gradienti
- fft_features        : Energia per bande di frequenza della FFT 2D
- statistical_moments : Media/Std/Skewness/Kurtosis su risposte Sobel
- laws_features       : Laws Texture Energy — 14 misure di energia locale
- extract_features    : Dispatcher — concatena i descrittori richiesti
- extract_batch       : Estrae feature da un intero batch di immagini (parallelizzato)
"""

import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.filters import gabor


# ────────────────────────────────────────────────────────────────────────────
# Funzioni statistiche ausiliarie (evitano overflow di scipy su array grandi)
# ────────────────────────────────────────────────────────────────────────────

def _skew64(x: np.ndarray) -> float:
    """
    Asimmetria (skewness) calcolata in float64 esplicito.
    Misura quanto la distribuzione è asimmetrica rispetto alla media.
    Valore positivo = coda destra, negativo = coda sinistra.
    """
    x = x.astype(np.float64, copy=False)
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-10:
        return 0.0
    return float(((x - mu) ** 3).mean() / sigma ** 3)


def _kurt64(x: np.ndarray) -> float:
    """
    Curtosi in eccesso (excess kurtosis) in float64.
    Misura quanto la distribuzione è "appuntita" rispetto a una gaussiana.
    Valore > 0 = code pesanti (anomalie), valore < 0 = distribuzione piatta.
    """
    x = x.astype(np.float64, copy=False)
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-10:
        return 0.0
    return float(((x - mu) ** 4).mean() / sigma ** 4) - 3.0


# ────────────────────────────────────────────────────────────────────────────
# Gabor filter bank
# ────────────────────────────────────────────────────────────────────────────

# 4 frequenze × 6 orientazioni = 24 filtri totali
GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4]
GABOR_THETAS = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]


def gabor_features(img: np.ndarray,
                   frequencies: list = GABOR_FREQUENCIES,
                   thetas: list = GABOR_THETAS) -> np.ndarray:
    """
    Bank di filtri Gabor: media e deviazione standard della risposta energetica
    per ogni combinazione (frequenza, orientazione).

    Un filtro Gabor è un'onda sinusoidale modulata da una gaussiana —
    sensibile a strutture periodiche a una certa scala e direzione.
    La parte reale rileva bordi, quella immaginaria le zone di transizione.

    Per ogni filtro: magnitude = sqrt(real² + imag²)
    Feature = [mean(magnitude), std(magnitude)] per ogni filtro
    → vettore di lunghezza 2 × 4 frequenze × 6 orientazioni = 48 valori

    Questo è uno dei descrittori più lenti (~24 convoluzioni per immagine).
    """
    feats = []
    for freq in frequencies:
        for theta in thetas:
            real, imag = gabor(img, frequency=freq, theta=theta)
            magnitude = np.sqrt(real.astype(np.float64) ** 2 +
                                imag.astype(np.float64) ** 2)
            feats.append(float(magnitude.mean()))
            feats.append(float(magnitude.std()))
    return np.array(feats, dtype=np.float64)


# ────────────────────────────────────────────────────────────────────────────
# GLCM — Gray-Level Co-occurrence Matrix
# ────────────────────────────────────────────────────────────────────────────

GLCM_DISTANCES = [1, 3]       # distanze tra coppie di pixel (in pixel)
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]   # 4 orientazioni
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def glcm_features(img: np.ndarray,
                  distances: list = GLCM_DISTANCES,
                  angles: list = GLCM_ANGLES,
                  levels: int = 64) -> np.ndarray:
    """
    Statistiche della Gray-Level Co-occurrence Matrix (GLCM).

    La GLCM conta quante volte una coppia di pixel con intensità i e j
    appaiono a una certa distanza e orientazione. Da essa si ricavano:
    - contrast     : variazione locale di intensità (alto su bordi forti)
    - dissimilarity: simile a contrast ma con peso lineare
    - homogeneity  : uniformità della texture (alto su zone lisce)
    - energy       : uniformità quadratica (alto su pattern regolari)
    - correlation  : quanto i pixel vicini sono linearmente correlati

    Riduce l'immagine a 64 livelli di grigio per rendere la GLCM
    trattabile (matrice 64×64 invece di 256×256) con dati limitati.

    Restituisce media e std di ogni proprietà → 10 valori totali.
    """
    # riduzione a 64 livelli: ogni bin rappresenta 4 livelli di grigio
    img_reduced = (img // (256 // levels)).astype(np.uint8)
    glcm = graycomatrix(img_reduced, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    feats = []
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)   # shape (len(distanze), len(angoli))
        feats.append(values.mean())        # media su tutte le distanze/angoli
        feats.append(values.std())         # variabilità direzionale
    return np.array(feats, dtype=np.float64)


# ────────────────────────────────────────────────────────────────────────────
# HOG — Histogram of Oriented Gradients
# ────────────────────────────────────────────────────────────────────────────

def hog_features(img: np.ndarray, pixels_per_cell: tuple = (16, 16),
                 cells_per_block: tuple = (2, 2),
                 orientations: int = 9) -> np.ndarray:
    """
    Histogram of Oriented Gradients.

    Divide l'immagine in celle di pixels_per_cell×pixels_per_cell pixel.
    Per ogni cella calcola l'istogramma delle orientazioni dei gradienti
    (9 bin da 0° a 180°). Le celle sono normalizzate in blocchi 2×2
    per robustezza alle variazioni di illuminazione.

    Resize fisso a 128×128 per garantire dimensione costante del vettore.
    Cattura la struttura dei bordi e la forma dei difetti.
    """
    img_resized = cv2.resize(img, (128, 128))
    feat = hog(img_resized,
               orientations=orientations,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block,
               feature_vector=True)
    return feat.astype(np.float64)


# ────────────────────────────────────────────────────────────────────────────
# FFT — Energia per bande di frequenza
# ────────────────────────────────────────────────────────────────────────────

def fft_features(img: np.ndarray, n_bands: int = 8) -> np.ndarray:
    """
    Energia dello spettro di Fourier 2D divisa in anelli concentrici.

    Lo spettro di magnitudine viene centrato sul DC (frequenza zero al centro).
    Viene diviso in n_bands anelli radiali di larghezza uguale.
    L'energia di ogni anello è una feature (normalizzata alla somma totale).

    Utile per texture periodiche (es. tile, grid): un difetto rompe
    la periodicità → distribuzione dell'energia cambia nelle bande.
    Meno efficace su texture stocastiche (carpet, legno con venatura irregolare).

    Restituisce vettore di 8 valori (energia relativa per banda).
    """
    f = np.fft.fft2(img.astype(np.float64))
    f_shifted = np.fft.fftshift(f)   # sposta DC al centro
    magnitude = np.abs(f_shifted)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # distanza radiale di ogni pixel dal centro
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    max_r = min(cx, cy)
    band_edges = np.linspace(0, max_r, n_bands + 1)

    feats = []
    for i in range(n_bands):
        mask = (r >= band_edges[i]) & (r < band_edges[i + 1])
        energy = magnitude[mask].sum()
        feats.append(energy)

    feats = np.array(feats, dtype=np.float64)
    total = feats.sum() + 1e-12
    return feats / total  # energia relativa (somma = 1)


# ────────────────────────────────────────────────────────────────────────────
# Statistical moments — momenti statistici su risposte di filtro
# ────────────────────────────────────────────────────────────────────────────

def statistical_moments(img: np.ndarray) -> np.ndarray:
    """
    Momenti statistici (media, std, skewness, kurtosis) applicati a:
    - pixel grezzi dell'immagine
    - risposta al filtro Sobel orizzontale (bordi verticali)
    - risposta al filtro Sobel verticale (bordi orizzontali)
    - magnitudine del gradiente Sobel (forza complessiva dei bordi)

    4 canali × 4 momenti = 16 valori totali.

    Questo descrittore è uno dei più efficaci su materiali con difetti
    macroscopici (wood, tile) perché i difetti alterano drasticamente
    la distribuzione del gradiente.
    """
    img_f = img.astype(np.float64)
    # filtri Sobel: derivate prime in x e y
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)   # bordi verticali
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)   # bordi orizzontali
    smag = np.sqrt(sx ** 2 + sy ** 2)                 # magnitudine gradiente

    feats = []
    for arr in [img_f, sx, sy, smag]:
        flat = arr.ravel().astype(np.float64)
        std_val = float(flat.std())
        if std_val < 1e-10:
            # immagine costante (es. zona nera uniforme) → momenti tutti zero
            feats.extend([float(flat.mean()), std_val, 0.0, 0.0])
        else:
            feats.extend([float(flat.mean()), std_val,
                          _skew64(flat), _kurt64(flat)])
    return np.array(feats, dtype=np.float64)


# ────────────────────────────────────────────────────────────────────────────
# Laws Texture Energy Measures (Laws, 1980)
# ────────────────────────────────────────────────────────────────────────────

def laws_features(img: np.ndarray) -> np.ndarray:
    """
    Laws Texture Energy: convoluzione con kernel 5×5 derivati da
    4 vettori base: L5 (livello), E5 (bordo), S5 (spot), R5 (ripple).

    I 16 kernel (prodotti esterni) catturano pattern texture diversi.
    Vengono calcolate 14 misure di energia sulle coppie simmetriche.
    Si esclude LL (solo componente di livello = media globale).

    Restituisce 14 valori di energia.
    """
    # vettori base di Laws
    L5 = np.array([1, 4, 6, 4, 1], dtype=np.float64)    # livello (smooth)
    E5 = np.array([-1, -2, 0, 2, 1], dtype=np.float64)  # bordo (edge)
    S5 = np.array([-1, 0, 2, 0, -1], dtype=np.float64)  # spot (blob)
    R5 = np.array([1, -4, 6, -4, 1], dtype=np.float64)  # ripple (onde)

    vectors = [L5, E5, S5, R5]
    names = ['L', 'E', 'S', 'R']

    img_f = img.astype(np.float64)
    # rimuove la componente DC (media) per focalizzarsi sulla texture
    img_f = img_f - img_f.mean()

    # calcola le 16 risposte ai filtri (prodotti 2D dei vettori)
    responses = {}
    for i, (v1, n1) in enumerate(zip(vectors, names)):
        for j, (v2, n2) in enumerate(zip(vectors, names)):
            kernel = np.outer(v1, v2)
            resp = cv2.filter2D(img_f, -1, kernel)
            responses[n1 + n2] = resp

    # 14 coppie simmetriche (AB e BA hanno stessa energia, si fanno la media)
    pairs = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i == 0 and j == 0:
                continue  # salta LL
            key = n1 + n2
            key_sym = n2 + n1
            pair = tuple(sorted([key, key_sym]))
            if pair not in pairs:
                pairs.append(pair)

    feats = []
    for p in pairs:
        if p[0] == p[1]:
            energy = np.mean(np.abs(responses[p[0]]))
        else:
            energy = np.mean(np.abs(responses[p[0]] + responses[p[1]])) / 2.0
        feats.append(energy)

    return np.array(feats, dtype=np.float64)


# ────────────────────────────────────────────────────────────────────────────
# Dispatcher — punto di ingresso principale
# ────────────────────────────────────────────────────────────────────────────

# Lista di tutti i nomi di feature disponibili (usata in pipeline.py)
AVAILABLE_FEATURES = ["gabor", "glcm", "hog", "fft", "stats", "laws"]

# Mappa nome → funzione estrattrice
_EXTRACTOR_MAP = {
    "gabor":   gabor_features,
    "glcm":    glcm_features,
    "hog":     hog_features,
    "fft":     fft_features,
    "stats":   statistical_moments,
    "laws":    laws_features,
}


def extract_features(img: np.ndarray,
                     feature_names: list = None) -> np.ndarray:
    """
    Estrae e concatena tutti i descrittori richiesti da una singola immagine.

    Gestisce overflow numerici (NaN, inf) sostituendoli con 0 —
    necessario perché alcune feature (es. kurtosis su immagini costanti)
    possono produrre valori indeterminati.

    Parametri
    ----------
    img           : immagine grayscale uint8 (H×W)
    feature_names : lista di nomi da AVAILABLE_FEATURES.
                    Se None, estrae tutte le feature.

    Ritorna
    -------
    vettore float64 1-D con tutte le feature concatenate
    """
    if feature_names is None:
        feature_names = AVAILABLE_FEATURES
    # np.errstate sopprime warning di overflow durante il calcolo
    with np.errstate(over='ignore', invalid='ignore'):
        parts = [_EXTRACTOR_MAP[name](img) for name in feature_names]
    vec = np.concatenate(parts).astype(np.float64)
    # sostituisce NaN e infiniti con 0 (robusto a immagini degeneri)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def extract_batch(images: np.ndarray,
                  feature_names: list = None,
                  verbose: bool = True,
                  n_jobs: int = -1) -> np.ndarray:
    """
    Estrae feature da un intero batch di immagini (N, H, W).

    Parallelizzato con joblib su tutti i core disponibili (n_jobs=-1).
    Usa thread (prefer="threads") invece di processi perché NumPy, OpenCV
    e skimage rilasciano il GIL durante le operazioni pesanti (convoluzione,
    FFT) — i thread si parallelizzano davvero senza overhead di memoria.

    Speedup atteso: da ~400s → ~60-80s su 8 core per 400 immagini.

    Ritorna
    -------
    X : array (N, D) con le feature di tutte le immagini
    """
    from joblib import Parallel, delayed

    def _extract_one(i, img):
        if verbose and (i % 20 == 0):
            print(f"  extracting features: {i}/{len(images)}")
        return extract_features(img, feature_names)

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_extract_one)(i, img) for i, img in enumerate(images)
    )
    return np.vstack(rows)
