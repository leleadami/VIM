"""
feature_extraction.py
---------------------
All texture feature extractors. Every function takes a single grayscale
uint8 image (H×W) and returns a 1-D float64 feature vector.

Extractors
----------
- lbp_features        : Local Binary Pattern histogram
- clbp_features       : Completed LBP (sign + magnitude + centre)
- gabor_features      : Gabor filter bank energy statistics
- glcm_features       : Gray-Level Co-occurrence Matrix statistics
- hog_features        : Histogram of Oriented Gradients
- fft_features        : FFT frequency-band energy
- wavelet_features    : Wavelet sub-band energy (pywt)
- statistical_moments : Mean/Std/Skewness/Kurtosis of filter responses
- extract_features    : Dispatcher — builds a named feature vector
"""

import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage.filters import gabor


def _skew64(x: np.ndarray) -> float:
    """Skewness in explicit float64 — avoids scipy overflow on large arrays."""
    x = x.astype(np.float64, copy=False)
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-10:
        return 0.0
    return float(((x - mu) ** 3).mean() / sigma ** 3)


def _kurt64(x: np.ndarray) -> float:
    """Excess kurtosis in explicit float64 — avoids scipy overflow."""
    x = x.astype(np.float64, copy=False)
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-10:
        return 0.0
    return float(((x - mu) ** 4).mean() / sigma ** 4) - 3.0


# ────────────────────────────────────────────────────────────────────────────
# LBP
# ────────────────────────────────────────────────────────────────────────────

def lbp_features(img: np.ndarray, P: int = 8, R: float = 1.0,
                 method: str = "uniform") -> np.ndarray:
    """
    Local Binary Pattern histogram.

    Parameters
    ----------
    P      : number of circularly symmetric neighbour set points
    R      : radius of circle
    method : 'uniform' | 'ror' (rotation-invariant)
    """
    lbp = local_binary_pattern(img, P, R, method=method)
    n_bins = P + 2 if method == "uniform" else P * (P - 1) + 3
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float64)


# ────────────────────────────────────────────────────────────────────────────
# Multi-scale LBP
# ────────────────────────────────────────────────────────────────────────────

def lbp_multiscale_features(img: np.ndarray,
                            scales: list = None,
                            method: str = "uniform") -> np.ndarray:
    """
    Multi-scale LBP: concatenates uniform LBP histograms at multiple
    (P, R) scales to capture texture at different spatial resolutions.

    Default scales: (P=8,R=1), (P=16,R=2), (P=24,R=3).
    Returns concatenated normalised histograms.
    """
    if scales is None:
        scales = [(8, 1.0), (16, 2.0), (24, 3.0)]
    hists = []
    for P, R in scales:
        lbp = local_binary_pattern(img, P, R, method=method)
        n_bins = P + 2 if method == "uniform" else P * (P - 1) + 3
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)
        hists.append(hist)
    return np.concatenate(hists).astype(np.float64)


# ────────────────────────────────────────────────────────────────────────────
# CLBP  (Completed LBP — Guo et al. 2010)
# ────────────────────────────────────────────────────────────────────────────

def clbp_features(img: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    """
    Completed LBP: concatenation of CLBP_S (sign), CLBP_M (magnitude),
    and CLBP_C (centre threshold) histograms.

    CLBP_S  — same bit-string as standard LBP (sign of differences)
    CLBP_M  — magnitude thresholded against mean magnitude of the image
    CLBP_C  — centre pixel thresholded against global mean

    All three histograms are individually normalised and concatenated.
    Reference: Guo et al., "A Completed Modeling of LBP", IEEE TIP 2010.
    """
    img_f = img.astype(np.float64)
    h, w = img_f.shape
    clbp_s = np.zeros((h, w), dtype=np.float64)
    clbp_m = np.zeros((h, w), dtype=np.float64)
    clbp_c = np.zeros((h, w), dtype=np.float64)

    angles = 2 * np.pi * np.arange(P) / P
    global_mean = img_f.mean()

    # accumulate magnitude over all neighbours
    diffs = np.zeros((h, w, P), dtype=np.float64)

    for idx, angle in enumerate(angles):
        x_off = R * np.cos(angle)
        y_off = -R * np.sin(angle)

        # bilinear interpolation
        x0 = int(np.floor(x_off))
        y0 = int(np.floor(y_off))
        dx = x_off - x0
        dy = y_off - y0

        # sample neighbour with boundary check
        def _safe(r_off, c_off):
            rr = np.clip(np.arange(h) + r_off, 0, h - 1).astype(int)
            cc = np.clip(np.arange(w) + c_off, 0, w - 1).astype(int)
            return img_f[np.ix_(rr, cc)]

        neighbour = ((1 - dx) * (1 - dy) * _safe(y0, x0) +
                     dx       * (1 - dy) * _safe(y0, x0 + 1) +
                     (1 - dx) * dy       * _safe(y0 + 1, x0) +
                     dx       * dy       * _safe(y0 + 1, x0 + 1))

        diff = neighbour - img_f
        diffs[:, :, idx] = diff
        clbp_s += (diff >= 0).astype(np.float64) * (2 ** idx)

    mean_magnitude = np.abs(diffs).mean()
    for idx in range(P):
        clbp_m += (np.abs(diffs[:, :, idx]) >= mean_magnitude).astype(np.float64) * (2 ** idx)

    clbp_c = (img_f >= global_mean).astype(np.float64)

    n_bins = 2 ** P
    hist_s, _ = np.histogram(clbp_s.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist_m, _ = np.histogram(clbp_m.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist_c = np.array([clbp_c.mean(), 1.0 - clbp_c.mean()])

    return np.concatenate([hist_s, hist_m, hist_c]).astype(np.float64)


# ────────────────────────────────────────────────────────────────────────────
# Gabor filter bank
# ────────────────────────────────────────────────────────────────────────────

GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4]
GABOR_THETAS = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]


def gabor_features(img: np.ndarray,
                   frequencies: list = GABOR_FREQUENCIES,
                   thetas: list = GABOR_THETAS) -> np.ndarray:
    """
    Gabor filter bank: mean and standard deviation of energy responses
    over all (frequency, orientation) pairs.

    Returns vector of length 2 × len(frequencies) × len(thetas).
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
# GLCM
# ────────────────────────────────────────────────────────────────────────────

GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def glcm_features(img: np.ndarray,
                  distances: list = GLCM_DISTANCES,
                  angles: list = GLCM_ANGLES,
                  levels: int = 64) -> np.ndarray:
    """
    Gray-Level Co-occurrence Matrix statistics.

    Reduces image to 64 grey levels to keep GLCM tractable and avoid
    sparse co-occurrence matrices with limited training data.
    Returns mean and std of each property over all (distance, angle) pairs.
    """
    img_reduced = (img // (256 // levels)).astype(np.uint8)
    glcm = graycomatrix(img_reduced, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    feats = []
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)   # shape (len(d), len(a))
        feats.append(values.mean())
        feats.append(values.std())
    return np.array(feats, dtype=np.float64)


# ────────────────────────────────────────────────────────────────────────────
# HOG
# ────────────────────────────────────────────────────────────────────────────

def hog_features(img: np.ndarray, pixels_per_cell: tuple = (16, 16),
                 cells_per_block: tuple = (2, 2),
                 orientations: int = 9) -> np.ndarray:
    """
    Histogram of Oriented Gradients.

    Resize to a fixed size first so the feature dimension is always constant.
    """
    img_resized = cv2.resize(img, (128, 128))
    feat = hog(img_resized,
               orientations=orientations,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block,
               feature_vector=True)
    return feat.astype(np.float64)


# ────────────────────────────────────────────────────────────────────────────
# FFT frequency-band energy
# ────────────────────────────────────────────────────────────────────────────

def fft_features(img: np.ndarray, n_bands: int = 8) -> np.ndarray:
    """
    Energy in concentric frequency bands of the 2D Fourier spectrum.

    The DC-centred magnitude spectrum is divided radially into `n_bands`
    annular rings of equal width; the energy in each ring is one feature.
    Returns a vector of length n_bands.
    """
    f = np.fft.fft2(img.astype(np.float64))
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
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
    return feats / total  # normalise to relative energy


# ────────────────────────────────────────────────────────────────────────────
# Wavelet sub-band energy
# ────────────────────────────────────────────────────────────────────────────

def wavelet_features(img: np.ndarray, wavelet: str = "db4",
                     level: int = 3) -> np.ndarray:
    """
    Discrete Wavelet Transform sub-band energy.

    Decomposes the image up to `level` levels using a 2D DWT.
    For each level, computes the energy (normalised L2-norm) of the
    detail sub-bands (LH, HL, HH). Also includes the approximation
    sub-band energy at the coarsest level.

    Returns vector of length 3*level + 1.
    """
    img_f = img.astype(np.float64)
    coeffs = pywt.wavedec2(img_f, wavelet=wavelet, level=level)

    feats = []
    # approximation at coarsest level
    cA = coeffs[0].astype(np.float64)
    feats.append(float(np.sqrt((cA ** 2).mean())))

    # detail sub-bands per level  (from coarsest to finest)
    for detail in coeffs[1:]:
        for band in detail:
            b = band.astype(np.float64)
            feats.append(float(np.sqrt((b ** 2).mean())))

    return np.array(feats, dtype=np.float64)


# ────────────────────────────────────────────────────────────────────────────
# Statistical moments on filter responses
# ────────────────────────────────────────────────────────────────────────────

def statistical_moments(img: np.ndarray) -> np.ndarray:
    """
    Per-channel descriptive statistics on raw pixel intensities plus first
    and second derivative (Sobel) magnitude maps.

    Returns: mean, std, skewness, kurtosis for each of:
        [raw, sobel_x, sobel_y, sobel_mag]  →  16 values
    """
    img_f = img.astype(np.float64)
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    smag = np.sqrt(sx ** 2 + sy ** 2)

    feats = []
    for arr in [img_f, sx, sy, smag]:
        flat = arr.ravel().astype(np.float64)
        std_val = float(flat.std())
        if std_val < 1e-10:
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
    Laws texture energy: convolve with 5x5 kernels from L5, E5, S5, R5
    vectors. Compute energy (mean abs) for each of 14 symmetric pairs.
    Returns 14 values.
    """
    L5 = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    E5 = np.array([-1, -2, 0, 2, 1], dtype=np.float64)
    S5 = np.array([-1, 0, 2, 0, -1], dtype=np.float64)
    R5 = np.array([1, -4, 6, -4, 1], dtype=np.float64)

    vectors = [L5, E5, S5, R5]
    names = ['L', 'E', 'S', 'R']

    img_f = img.astype(np.float64)
    # Remove DC component
    img_f = img_f - img_f.mean()

    # Compute all 16 filter responses
    responses = {}
    for i, (v1, n1) in enumerate(zip(vectors, names)):
        for j, (v2, n2) in enumerate(zip(vectors, names)):
            kernel = np.outer(v1, v2)
            resp = cv2.filter2D(img_f, -1, kernel)
            responses[n1 + n2] = resp

    # 14 symmetric energy features (excluding LL)
    pairs = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i == 0 and j == 0:
                continue  # skip LL
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
# Dense SIFT + Bag of Visual Words (BoVW)
# ────────────────────────────────────────────────────────────────────────────

# Global codebook (built lazily at first call)
_bovw_codebook = None
_bovw_n_words = 64

def _build_codebook(descriptors: np.ndarray, n_words: int = 64):
    """Build KMeans codebook from SIFT descriptors."""
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_words, random_state=42,
                             batch_size=1000, max_iter=100)
    kmeans.fit(descriptors)
    return kmeans

def dsift_bovw_features(img: np.ndarray) -> np.ndarray:
    """
    Dense SIFT + Bag of Visual Words.
    Extract SIFT on a dense grid, encode as histogram of visual words.
    Returns histogram of n_words bins (default 64).
    """
    global _bovw_codebook
    sift = cv2.SIFT_create()
    h, w = img.shape[:2]
    step = 8
    kps = [cv2.KeyPoint(float(x), float(y), float(step))
           for y in range(step, h - step, step)
           for x in range(step, w - step, step)]

    _, descriptors = sift.compute(img, kps)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(_bovw_n_words, dtype=np.float64)

    # Build codebook on first call (from this image's descriptors as seed)
    if _bovw_codebook is None:
        _bovw_codebook = _build_codebook(descriptors, _bovw_n_words)

    # Encode: assign each descriptor to nearest word, build histogram
    words = _bovw_codebook.predict(descriptors)
    hist = np.bincount(words, minlength=_bovw_n_words).astype(np.float64)
    # L1 normalize
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# ────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────────────────────

AVAILABLE_FEATURES = ["lbp", "lbp_ms", "clbp", "gabor", "glcm", "hog",
                      "fft", "wavelet", "stats", "laws", "dsift"]

_EXTRACTOR_MAP = {
    "lbp":     lbp_features,
    "lbp_ms":  lbp_multiscale_features,
    "clbp":    clbp_features,
    "gabor":   gabor_features,
    "glcm":    glcm_features,
    "hog":     hog_features,
    "fft":     fft_features,
    "wavelet": wavelet_features,
    "stats":   statistical_moments,
    "laws":    laws_features,
    "dsift":   dsift_bovw_features,
}


def extract_features(img: np.ndarray,
                     feature_names: list = None) -> np.ndarray:
    """
    Extract and concatenate all requested feature descriptors from a single
    grayscale image.

    Parameters
    ----------
    img           : grayscale uint8 numpy array
    feature_names : list of str — subset of AVAILABLE_FEATURES.
                    If None, all features are extracted.

    Returns
    -------
    feature_vector : 1-D float64 array
    """
    if feature_names is None:
        feature_names = AVAILABLE_FEATURES
    with np.errstate(over='ignore', invalid='ignore'):
        parts = [_EXTRACTOR_MAP[name](img) for name in feature_names]
    vec = np.concatenate(parts).astype(np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def extract_batch(images: np.ndarray,
                  feature_names: list = None,
                  verbose: bool = True) -> np.ndarray:
    """
    Extract features from an (N, H, W) image array.

    Returns
    -------
    X : np.ndarray of shape (N, D)
    """
    rows = []
    for i, img in enumerate(images):
        if verbose and (i % 20 == 0):
            print(f"  extracting features: {i}/{len(images)}")
        rows.append(extract_features(img, feature_names))
    return np.vstack(rows)
