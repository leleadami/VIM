"""
models.py
---------
Anomaly detectors and classifiers.

Unsupervised (trained on normal samples only):
  - OneClassSVM
  - IsolationForest
  - LocalOutlierFactor
  - GMMDetector  (Gaussian Mixture Model + Mahalanobis distance)
  - EllipticEnvelope
  - PCANullSubspace  (PCA-based anomaly score without any classifier)

Supervised baselines (trained with both labels):
  - SVMClassifier
  - RandomForestClassifier

All detectors expose a common interface:
    .fit(X_normal)
    .score_samples(X)   → anomaly score (higher = more anomalous)
    .predict(X)         → binary labels  1=anomaly, 0=normal

Supervised classifiers expose:
    .fit(X, y)
    .predict(X)
    .predict_proba(X)
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EllipticEnvelope as _EE
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest as _IF
from sklearn.ensemble import RandomForestClassifier as _RF
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor as _LOF
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM as _OCSVM
from sklearn.svm import SVC


# ────────────────────────────────────────────────────────────────────────────
# Base class
# ────────────────────────────────────────────────────────────────────────────

class BaseDetector:
    """Minimal interface shared by all unsupervised detectors."""

    def fit(self, X: np.ndarray):
        raise NotImplementedError

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly score per sample (higher = more anomalous)."""
        raise NotImplementedError

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Binary predictions.  By default uses the 95th percentile of training
        scores as threshold (set during fit).  Pass threshold to override.
        """
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._threshold
        return (scores >= thr).astype(int)


# ────────────────────────────────────────────────────────────────────────────
# PCA pre-processing wrapper (shared by multiple detectors)
# ────────────────────────────────────────────────────────────────────────────

class _PCAPreprocessor:
    def __init__(self, n_components: float = 0.95):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit_transform(self, X):
        X_s = self.scaler.fit_transform(X)
        return self.pca.fit_transform(X_s)

    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X))


# ────────────────────────────────────────────────────────────────────────────
# One-Class SVM
# ────────────────────────────────────────────────────────────────────────────

class OneClassSVMDetector(BaseDetector):
    """
    One-Class SVM with RBF kernel.
    Features are standardised + PCA-reduced before fitting.
    """

    def __init__(self, nu: float = 0.1, kernel: str = "rbf",
                 pca_variance: float = 0.95):
        self._pre = _PCAPreprocessor(pca_variance)
        self._model = _OCSVM(nu=nu, kernel=kernel)
        self._threshold = None

    def fit(self, X: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._model.fit(Xp)
        raw_scores = -self._model.decision_function(Xp)
        self._threshold = np.percentile(raw_scores, 95)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pre.transform(X)
        return -self._model.decision_function(Xp)


# ────────────────────────────────────────────────────────────────────────────
# Isolation Forest
# ────────────────────────────────────────────────────────────────────────────

class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest anomaly detector.
    Returns negated sklearn score so higher = more anomalous.
    """

    def __init__(self, n_estimators: int = 200, contamination: float = 0.05,
                 pca_variance: float = 0.95):
        self._pre = _PCAPreprocessor(pca_variance)
        self._model = _IF(n_estimators=n_estimators,
                          contamination=contamination,
                          random_state=42,
                          n_jobs=-1)
        self._threshold = None

    def fit(self, X: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._model.fit(Xp)
        raw_scores = -self._model.score_samples(Xp)
        self._threshold = np.percentile(raw_scores, 95)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pre.transform(X)
        return -self._model.score_samples(Xp)


# ────────────────────────────────────────────────────────────────────────────
# Local Outlier Factor
# ────────────────────────────────────────────────────────────────────────────

class LOFDetector(BaseDetector):
    """
    Local Outlier Factor.
    Note: LOF in sklearn has novelty=True required for predict on new data.
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05,
                 pca_variance: float = 0.95):
        self._pre = _PCAPreprocessor(pca_variance)
        self._model = _LOF(n_neighbors=n_neighbors,
                           contamination=contamination,
                           novelty=True,
                           n_jobs=-1)
        self._threshold = None

    def fit(self, X: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._model.fit(Xp)
        raw_scores = -self._model.score_samples(Xp)
        self._threshold = np.percentile(raw_scores, 95)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pre.transform(X)
        return -self._model.score_samples(Xp)


# ────────────────────────────────────────────────────────────────────────────
# GMM + Mahalanobis
# ────────────────────────────────────────────────────────────────────────────

class GMMDetector(BaseDetector):
    """
    Gaussian Mixture Model trained on normal features.
    Anomaly score = negative log-likelihood.

    Optionally also computes Mahalanobis distance to the nearest Gaussian
    component mean (disabled by default — GMM NLL is sufficient).
    """

    def __init__(self, n_components: int = 3, covariance_type: str = "full",
                 pca_variance: float = 0.95):
        self._pre = _PCAPreprocessor(pca_variance)
        self._model = GaussianMixture(n_components=n_components,
                                      covariance_type=covariance_type,
                                      random_state=42,
                                      max_iter=300)
        self._threshold = None

    def fit(self, X: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._model.fit(Xp)
        raw_scores = -self._model.score_samples(Xp)
        self._threshold = np.percentile(raw_scores, 95)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pre.transform(X)
        return -self._model.score_samples(Xp)


# ────────────────────────────────────────────────────────────────────────────
# Elliptic Envelope
# ────────────────────────────────────────────────────────────────────────────

class EllipticEnvelopeDetector(BaseDetector):
    """
    Robust covariance estimation + Mahalanobis distance thresholding.
    Assumes a single multivariate Gaussian for normal features.
    Best suited to unimodal feature distributions (e.g. single texture category).
    """

    def __init__(self, contamination: float = 0.05, pca_variance: float = 0.95):
        self._pre = _PCAPreprocessor(pca_variance)
        self._model = _EE(contamination=contamination,
                          support_fraction=0.9,
                          random_state=42)
        self._threshold = None

    def fit(self, X: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._model.fit(Xp)
        raw_scores = -self._model.score_samples(Xp)
        self._threshold = np.percentile(raw_scores, 95)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pre.transform(X)
        return -self._model.score_samples(Xp)


# ────────────────────────────────────────────────────────────────────────────
# PCA Null Subspace
# ────────────────────────────────────────────────────────────────────────────

class PCANullSubspaceDetector(BaseDetector):
    """
    PCA Null Subspace anomaly detector.

    Fit PCA on normal features; retain only the eigenvectors corresponding
    to the *lowest* variance directions (null subspace).

    Anomaly score = L2 norm of the projection of (x - μ) onto the null
    subspace:
        score(x) = || W_low^T (x - μ) ||_2

    where W_low spans the eigenvectors EXCLUDED by a standard PCA that
    retains `retained_variance` fraction of total variance.

    Intuition: normal samples lie close to the principal subspace; anomalies
    "leak" energy into the null subspace.

    Reference: Null Subspace PCA Cascade (PMC:12349016, 2025).
    """

    def __init__(self, retained_variance: float = 0.95):
        self.scaler = StandardScaler()
        self.retained_variance = retained_variance
        self._W_low = None
        self._mean = None
        self._threshold = None

    def fit(self, X: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self._mean = Xs.mean(axis=0)

        # Full PCA on normal features
        pca_full = PCA().fit(Xs)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, self.retained_variance)) + 1
        # W_low: columns are the low-variance eigenvectors
        self._W_low = pca_full.components_[k:].T   # shape (D, n_null)

        scores = self._score_raw(Xs)
        self._threshold = np.percentile(scores, 95)
        return self

    def _score_raw(self, Xs: np.ndarray) -> np.ndarray:
        centred = Xs - self._mean
        proj = centred @ self._W_low          # (N, n_null)
        return np.linalg.norm(proj, axis=1)   # (N,)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self._score_raw(Xs)


# ────────────────────────────────────────────────────────────────────────────
# Supervised classifiers
# ────────────────────────────────────────────────────────────────────────────

class SVMClassifier:
    """Binary SVM with probability estimates (supervised baseline)."""

    def __init__(self, C: float = 1.0, kernel: str = "rbf",
                 pca_variance: float = 0.95):
        self._pre = _PCAPreprocessor(pca_variance)
        self._model = SVC(C=C, kernel=kernel, probability=True,
                          class_weight="balanced", random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._model.fit(Xp, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._pre.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._pre.transform(X))[:, 1]


class RandomForestDetector:
    """Binary Random Forest (supervised baseline)."""

    def __init__(self, n_estimators: int = 200, pca_variance: float = 0.95):
        self._pre = _PCAPreprocessor(pca_variance)
        self._model = _RF(n_estimators=n_estimators,
                          class_weight="balanced",
                          random_state=42,
                          n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._model.fit(Xp, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._pre.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._pre.transform(X))[:, 1]


# ────────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────────

UNSUPERVISED_DETECTORS = {
    "OC-SVM":           OneClassSVMDetector,
    "IsolationForest":  IsolationForestDetector,
    "LOF":              LOFDetector,
    "GMM":              GMMDetector,
    "EllipticEnvelope": EllipticEnvelopeDetector,
    "PCANullSubspace":  PCANullSubspaceDetector,
}

SUPERVISED_CLASSIFIERS = {
    "SVM":          SVMClassifier,
    "RandomForest": RandomForestDetector,
}
