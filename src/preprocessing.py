"""
preprocessing.py
----------------
Image denoising and contrast enhancement utilities.
All functions accept and return grayscale uint8 numpy arrays.

Denoising methods:
  - Gaussian, Median, Bilateral (basic)
  - Non-Local Means (NLMeans) — edge-preserving, effective at high SNR
  - Wavelet thresholding (BayesShrink) — preserves texture edges at low SNR
  - NLMeans + Wavelet (RCLBP approach) — robust combination from
    Gyimah et al. (2021), arXiv:2112.04021
"""

import cv2
import numpy as np
import pywt


# ── Denoising ───────────────────────────────────────────────────────────────

def gaussian_denoise(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian low-pass filter for additive Gaussian noise."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filter — robust to salt-and-pepper noise."""
    return cv2.medianBlur(img, ksize)


def bilateral_denoise(img: np.ndarray, d: int = 9,
                      sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Bilateral filter — edge-preserving smoothing."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def nlmeans_denoise(img: np.ndarray, h: float = 10,
                    template_window: int = 7,
                    search_window: int = 21) -> np.ndarray:
    """
    Non-Local Means denoising (cv2.fastNlMeansDenoising).

    Preserves edges and texture structure better than Gaussian/median
    at moderate noise levels. Key building block of the RCLBP framework
    (Gyimah et al., 2021).

    Parameters
    ----------
    h               : filter strength (higher = more denoising, less detail)
    template_window : size of template patch (should be odd)
    search_window   : size of the search area (should be odd)
    """
    return cv2.fastNlMeansDenoising(img, None, h,
                                     template_window, search_window)


def wavelet_denoise(img: np.ndarray, wavelet: str = "db4",
                    level: int = 3, mode: str = "soft") -> np.ndarray:
    """
    Wavelet-domain denoising with BayesShrink adaptive thresholding.

    Decomposes the image into wavelet sub-bands, estimates a noise-adaptive
    threshold for each detail sub-band using the BayesShrink rule:
        T = sigma^2 / sigma_x
    where sigma is the noise standard deviation (estimated from the finest
    HH sub-band via the Median Absolute Deviation) and sigma_x is the
    signal standard deviation in each sub-band.

    Reference: Chang, Yu, Vetterli — "Adaptive wavelet thresholding for
    image denoising and compression", IEEE TIP, 2000.
    """
    img_f = img.astype(np.float64)
    coeffs = pywt.wavedec2(img_f, wavelet=wavelet, level=level)

    # Estimate noise sigma from the finest HH sub-band (MAD estimator)
    detail_coeffs = coeffs[-1]
    sigma_noise = np.median(np.abs(detail_coeffs[2])) / 0.6745

    # Apply BayesShrink to each detail sub-band
    new_coeffs = [coeffs[0]]  # keep approximation unchanged
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
    # Clip and convert back to uint8
    return np.clip(reconstructed[:img.shape[0], :img.shape[1]],
                   0, 255).astype(np.uint8)


def rclbp_denoise(img: np.ndarray, h: float = 10,
                  wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """
    RCLBP denoising: NLMeans + wavelet thresholding (Gyimah et al., 2021).

    1. Apply NLMeans to get I_F (filtered image).
    2. Compute method noise MN = V - I_F (lost textures/edges).
    3. Wavelet-threshold MN to recover clean texture detail D_hat.
    4. Return B = I_F + D_hat (denoised with restored edges).

    This combination is robust: NLMeans removes bulk noise while wavelet
    thresholding recovers texture details that NLMeans discards at low SNR.
    """
    img_f = img.astype(np.float64)
    I_F = nlmeans_denoise(img).astype(np.float64)

    # Method noise: difference contains lost texture + residual noise
    MN = img_f - I_F

    # Wavelet-threshold the method noise to recover clean texture detail
    D_hat = wavelet_denoise(
        np.clip(MN + 128, 0, 255).astype(np.uint8),
        wavelet=wavelet, level=level
    ).astype(np.float64) - 128.0

    # Reconstruct: filtered image + recovered texture detail
    B = I_F + D_hat
    return np.clip(B, 0, 255).astype(np.uint8)


# ── Contrast enhancement ─────────────────────────────────────────────────────

def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization."""
    return cv2.equalizeHist(img)


def clahe(img: np.ndarray, clip_limit: float = 2.0,
          tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Preferred over global HE for textured surfaces: avoids over-amplifying
    uniform regions while boosting local contrast in structured areas.
    """
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return c.apply(img)


# ── Combined preprocessing ────────────────────────────────────────────────────

def preprocess(img: np.ndarray,
               denoise: str = "gaussian",
               enhance: str = "clahe",
               **kwargs) -> np.ndarray:
    """
    Apply a denoising step followed by a contrast enhancement step.

    Parameters
    ----------
    img     : grayscale uint8 image
    denoise : 'gaussian' | 'median' | 'bilateral' | 'nlmeans' | 'wavelet'
              | 'rclbp' | None
    enhance : 'clahe' | 'histeq' | None
    """
    # -- denoising
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

    # -- contrast enhancement
    if enhance == "clahe":
        img = clahe(img)
    elif enhance == "histeq":
        img = histogram_equalization(img)

    return img


def preprocess_batch(images: np.ndarray, **kwargs) -> np.ndarray:
    """Apply preprocess() to every image in an (N, H, W) array."""
    return np.array([preprocess(img, **kwargs) for img in images])
