"""
feature_extraction.py
---------------------
Texture feature extractors for anomaly detection.
Each function takes a single grayscale uint8 image (H×W) and returns
a 1-D float64 feature vector.

Available descriptors
---------------------
- gabor_features      : Gabor filter bank — frequency/orientation analysis (48-D)
- glcm_features       : Gray-Level Co-occurrence Matrix statistics (10-D)
- hog_features        : Histogram of Oriented Gradients (1764-D)
- fft_features        : FFT radial band energy (8-D)
- statistical_moments : Sobel gradient moments — mean/std/skewness/kurtosis (16-D)
- laws_features       : Laws Texture Energy measures (14-D)
- extract_features    : dispatcher — concatenates requested descriptors
- extract_batch       : extracts features from a full image batch (parallelised)
"""

import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.filters import gabor


# ── Numerical helpers ─────────────────────────────────────────────────────────
# Manual float64 implementations avoid scipy dependency and float32 overflow
# that can occur when cubing/quarting large pixel values.

def _skew64(x: np.ndarray) -> float:
    """Third standardised moment (skewness) in float64."""
    x = x.astype(np.float64, copy=False)
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-10:
        return 0.0
    return float(((x - mu) ** 3).mean() / sigma ** 3)


def _kurt64(x: np.ndarray) -> float:
    """Excess kurtosis (fourth standardised moment minus 3) in float64."""
    x = x.astype(np.float64, copy=False)
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-10:
        return 0.0
    return float(((x - mu) ** 4).mean() / sigma ** 4) - 3.0


# ── Gabor filter bank ─────────────────────────────────────────────────────────

# 4 frequencies × 6 orientations = 24 filters → 48-D feature vector
GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4]
GABOR_THETAS = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]


def gabor_features(img: np.ndarray,
                   frequencies: list = GABOR_FREQUENCIES,
                   thetas: list = GABOR_THETAS) -> np.ndarray:
    """
    Gabor filter bank: mean and std of response magnitude per filter.

    Each Gabor filter is a sinusoidal plane wave modulated by a Gaussian
    envelope, tuned to a specific frequency and orientation. The magnitude
    response captures texture energy at that scale and direction.

    Output: [mean, std] × 24 filters = 48-D vector.
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


# ── GLCM ──────────────────────────────────────────────────────────────────────

GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def glcm_features(img: np.ndarray,
                  distances: list = GLCM_DISTANCES,
                  angles: list = GLCM_ANGLES,
                  levels: int = 64) -> np.ndarray:
    """
    Gray-Level Co-occurrence Matrix (GLCM) statistics (Haralick et al., 1973).

    Computes five texture properties from the GLCM at two distances and four
    angles, then summarises each as mean and std across all (d, theta) pairs.

    Image is quantised to 64 grey levels to keep the co-occurrence matrix
    tractable with limited training data.

    Output: 5 properties × 2 statistics = 10-D vector.
    """
    img_reduced = (img // (256 // levels)).astype(np.uint8)
    glcm = graycomatrix(img_reduced, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    feats = []
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)
        feats.append(values.mean())
        feats.append(values.std())
    return np.array(feats, dtype=np.float64)


# ── HOG ───────────────────────────────────────────────────────────────────────

def hog_features(img: np.ndarray, pixels_per_cell: tuple = (16, 16),
                 cells_per_block: tuple = (2, 2),
                 orientations: int = 9) -> np.ndarray:
    """
    Histogram of Oriented Gradients (Dalal & Triggs, 2005).

    Image is resized to 128×128 before extraction to fix the output
    dimension regardless of input size.
    Configuration: 16×16 cells, 2×2 block normalisation, 9 orientation bins.

    Output: 1764-D vector  (49 blocks × 4 cells × 9 bins).
    """
    img_resized = cv2.resize(img, (128, 128))
    feat = hog(img_resized,
               orientations=orientations,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block,
               feature_vector=True)
    return feat.astype(np.float64)


# ── FFT ───────────────────────────────────────────────────────────────────────

def fft_features(img: np.ndarray, n_bands: int = 8) -> np.ndarray:
    """
    Radial band energy of the 2D Fourier magnitude spectrum.

    The DC-centred magnitude spectrum is divided into n_bands concentric
    annular rings of equal radial width. Energy per band is normalised to
    sum to 1, making the descriptor invariant to global brightness.

    Effective on periodic textures (tile, grid) where defects shift the
    spectral energy distribution across bands.

    Output: 8-D normalised energy vector.
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
    return feats / total


# ── Statistical moments ───────────────────────────────────────────────────────

def statistical_moments(img: np.ndarray) -> np.ndarray:
    """
    Statistical moments of raw pixel values and Sobel gradient responses.

    Channels: raw image, Sobel-x, Sobel-y, gradient magnitude.
    Moments: mean, std, skewness, kurtosis  (4 channels × 4 moments = 16-D).

    Gradient kurtosis is particularly diagnostic: localised defects
    (holes, scratches) create heavy-tailed gradient distributions with
    kurtosis significantly above the normal baseline.
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
            # uniform patch (e.g. black border) — all higher moments are zero
            feats.extend([float(flat.mean()), std_val, 0.0, 0.0])
        else:
            feats.extend([float(flat.mean()), std_val,
                          _skew64(flat), _kurt64(flat)])
    return np.array(feats, dtype=np.float64)


# ── Laws Texture Energy ───────────────────────────────────────────────────────

def laws_features(img: np.ndarray) -> np.ndarray:
    """
    Laws Texture Energy Measures (Laws, 1980).

    Convolves the image with 5×5 kernels built from the outer products of
    four 1-D basis vectors: L5 (level), E5 (edge), S5 (spot), R5 (ripple).
    Energy is computed for 14 symmetric filter pairs (LL excluded as it
    captures only mean intensity, not texture).

    DC component is removed before filtering to focus on texture structure.

    Output: 14-D energy vector.
    """
    L5 = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    E5 = np.array([-1, -2, 0, 2, 1], dtype=np.float64)
    S5 = np.array([-1, 0, 2, 0, -1], dtype=np.float64)
    R5 = np.array([1, -4, 6, -4, 1], dtype=np.float64)

    vectors = [L5, E5, S5, R5]
    names = ['L', 'E', 'S', 'R']

    img_f = img.astype(np.float64)
    img_f = img_f - img_f.mean()  # remove DC component

    responses = {}
    for i, (v1, n1) in enumerate(zip(vectors, names)):
        for j, (v2, n2) in enumerate(zip(vectors, names)):
            kernel = np.outer(v1, v2)
            resp = cv2.filter2D(img_f, -1, kernel)
            responses[n1 + n2] = resp

    # collect 14 symmetric pairs (AB and BA share the same energy)
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


# ── Dispatcher ────────────────────────────────────────────────────────────────

AVAILABLE_FEATURES = ["gabor", "glcm", "hog", "fft", "stats", "laws"]

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
    Extract and concatenate the requested descriptors from a single image.

    NaN and infinite values (e.g. kurtosis on a constant-intensity patch)
    are replaced with 0 to prevent propagation into downstream models.

    Parameters
    ----------
    img           : grayscale uint8 image (H×W)
    feature_names : names from AVAILABLE_FEATURES; None = all features

    Returns
    -------
    1-D float64 vector of concatenated features
    """
    if feature_names is None:
        feature_names = AVAILABLE_FEATURES
    with np.errstate(over='ignore', invalid='ignore'):
        parts = [_EXTRACTOR_MAP[name](img) for name in feature_names]
    vec = np.concatenate(parts).astype(np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def extract_batch(images: np.ndarray,
                  feature_names: list = None,
                  verbose: bool = True,
                  n_jobs: int = -1) -> np.ndarray:
    """
    Extract features from a batch of images (N, H, W), parallelised.

    Uses threads rather than processes: NumPy, OpenCV and skimage release
    the GIL during heavy numerical operations (convolution, FFT), so
    thread-based parallelism achieves near-linear speedup without the
    memory overhead of multiprocessing.

    Returns
    -------
    X : (N, D) float64 feature matrix
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
