"""
preprocessing.py
----------------
Image denoising and contrast enhancement.
All functions accept and return grayscale uint8 numpy arrays.

Pipeline: denoise → enhance (order matters — denoising before contrast
enhancement avoids amplifying noise along with texture).

Denoising options : gaussian | median | bilateral | nlmeans | wavelet | rclbp
Enhancement options: clahe | histeq
"""

import cv2
import numpy as np
import pywt


# ── Denoising ────────────────────────────────────────────────────────────────

def gaussian_denoise(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian low-pass filter. Fast default for mild additive noise."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filter. Robust to salt-and-pepper noise; preserves edges."""
    return cv2.medianBlur(img, ksize)


def bilateral_denoise(img: np.ndarray, d: int = 9,
                      sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Bilateral filter — smooths flat regions while preserving edges."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def nlmeans_denoise(img: np.ndarray, h: float = 10,
                    template_window: int = 7,
                    search_window: int = 21) -> np.ndarray:
    """
    Non-Local Means denoising (Buades et al., 2005).

    Weights each pixel as a weighted average of similar patches in a
    search window. Better texture preservation than Gaussian or median
    at the cost of higher runtime.

    Parameters
    ----------
    h               : filter strength (higher = more smoothing, less detail)
    template_window : patch size for similarity comparison (odd)
    search_window   : search area radius (odd)
    """
    return cv2.fastNlMeansDenoising(img, None, h,
                                     template_window, search_window)


def wavelet_denoise(img: np.ndarray, wavelet: str = "db4",
                    level: int = 3, mode: str = "soft") -> np.ndarray:
    """
    Wavelet-domain denoising with BayesShrink adaptive thresholding.

    Estimates noise sigma from the finest HH sub-band via MAD, then
    applies a per-sub-band soft threshold:
        T = sigma_noise^2 / sigma_signal

    Reference: Chang, Yu, Vetterli — IEEE TIP 2000.
    """
    img_f = img.astype(np.float64)
    coeffs = pywt.wavedec2(img_f, wavelet=wavelet, level=level)

    # noise estimate from finest diagonal sub-band (robust to outliers)
    detail_coeffs = coeffs[-1]
    sigma_noise = np.median(np.abs(detail_coeffs[2])) / 0.6745

    new_coeffs = [coeffs[0]]  # keep approximation sub-band unchanged
    for detail in coeffs[1:]:
        new_detail = []
        for subband in detail:
            sigma_y_sq = np.mean(subband ** 2)
            sigma_x = np.sqrt(max(sigma_y_sq - sigma_noise ** 2, 0))
            if sigma_x == 0:
                threshold = np.max(np.abs(subband))
            else:
                threshold = sigma_noise ** 2 / sigma_x
            if mode == "soft":
                denoised = pywt.threshold(subband, threshold, mode="soft")
            else:
                denoised = pywt.threshold(subband, threshold, mode="hard")
            new_detail.append(denoised)
        new_coeffs.append(tuple(new_detail))

    reconstructed = pywt.waverec2(new_coeffs, wavelet=wavelet)
    return np.clip(reconstructed[:img.shape[0], :img.shape[1]],
                   0, 255).astype(np.uint8)


def rclbp_denoise(img: np.ndarray, h: float = 10,
                  wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """
    RCLBP denoising: NLMeans + wavelet thresholding (Gyimah et al., 2021).

    1. Apply NLMeans → I_F
    2. Compute method noise MN = V - I_F  (edges/texture discarded by NLM)
    3. Wavelet-threshold MN → clean detail D_hat
    4. Reconstruct B = I_F + D_hat

    Reference: arXiv:2112.04021
    """
    img_f = img.astype(np.float64)
    I_F = nlmeans_denoise(img).astype(np.float64)

    MN = img_f - I_F

    # shift MN to [0,255] range before wavelet processing, then re-centre
    D_hat = wavelet_denoise(
        np.clip(MN + 128, 0, 255).astype(np.uint8),
        wavelet=wavelet, level=level
    ).astype(np.float64) - 128.0

    B = I_F + D_hat
    return np.clip(B, 0, 255).astype(np.uint8)


# ── Contrast enhancement ─────────────────────────────────────────────────────

def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization."""
    return cv2.equalizeHist(img)


def clahe(img: np.ndarray, clip_limit: float = 2.0,
          tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Preferred over global HE for textured surfaces: enhances local
    contrast per tile without over-amplifying uniform regions.
    """
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return c.apply(img)


# ── Combined pipeline ─────────────────────────────────────────────────────────

def preprocess(img: np.ndarray,
               denoise: str = "gaussian",
               enhance: str = "clahe",
               **kwargs) -> np.ndarray:
    """
    Apply denoising followed by contrast enhancement.

    Parameters
    ----------
    img     : grayscale uint8 image
    denoise : 'gaussian' | 'median' | 'bilateral' | 'nlmeans' | 'wavelet'
              | 'rclbp' | None
    enhance : 'clahe' | 'histeq' | None
    """
    if denoise == "gaussian":
        img = gaussian_denoise(img, **{k: v for k, v in kwargs.items()
                                       if k in ("ksize", "sigma")})
    elif denoise == "median":
        img = median_denoise(img, **{k: v for k, v in kwargs.items()
                                     if k in ("ksize",)})
    elif denoise == "bilateral":
        img = bilateral_denoise(img)
    elif denoise == "nlmeans":
        img = nlmeans_denoise(img)
    elif denoise == "wavelet":
        img = wavelet_denoise(img)
    elif denoise == "rclbp":
        img = rclbp_denoise(img)

    if enhance == "clahe":
        img = clahe(img)
    elif enhance == "histeq":
        img = histogram_equalization(img)

    return img


def preprocess_batch(images: np.ndarray, **kwargs) -> np.ndarray:
    """
    Apply preprocess() to every image in a batch, parallelised with threads.

    Threads (not processes) work well here because OpenCV and NumPy release
    the GIL during convolution and FFT operations.
    """
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(preprocess)(img, **kwargs) for img in images
    )
    return np.array(results)
