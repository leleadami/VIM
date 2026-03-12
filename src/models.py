"""
models.py
---------
Anomaly detectors and classifiers.

Unsupervised (trained on normal samples only):
  - OneClassSVM
  - IsolationForest
  - LocalOutlierFactor
  - GMMDetector  (Gaussian Mixture Model + negative log-likelihood)
  - EllipticEnvelope
  - PCANullSubspace  (PCA-based anomaly score without any classifier)

Supervised baselines (trained with both labels):
  - SVMClassifier
  - RandomForestClassifier

All detectors expose a common interface:
    .fit(X_normal)
    .score_samples(X)   -> anomaly score (higher = more anomalous)
    .predict(X)         -> binary labels  1=anomaly, 0=normal

Supervised classifiers expose:
    .fit(X, y)
    .predict(X)
    .predict_proba(X)
"""

import numpy as np
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
# Generic sklearn-based detector (covers OC-SVM, IF, LOF, GMM, EE)
# ────────────────────────────────────────────────────────────────────────────

class SklearnDetector:
    """
    Wraps any sklearn unsupervised model that provides a scoring function.
    Applies StandardScaler + PCA before the model, and negates the sklearn
    score so that higher = more anomalous.

    Parameters
    ----------
    model       : sklearn estimator instance (already configured)
    score_fn    : name of the scoring method ('score_samples' or 'decision_function')
    pca_variance: fraction of variance retained by PCA (default 0.95)
    """

    def __init__(self, model, score_fn: str = "score_samples",
                 pca_variance: float = 0.95):
        self._pre = _PCAPreprocessor(pca_variance)
        self._model = model
        self._score_fn = score_fn
        self._threshold = None

    def fit(self, X: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._model.fit(Xp)
        raw_scores = -getattr(self._model, self._score_fn)(Xp)
        self._threshold = np.percentile(raw_scores, 90)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pre.transform(X)
        return -getattr(self._model, self._score_fn)(Xp)

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._threshold
        return (scores >= thr).astype(int)


# Factory functions — one per detector, so the registry stays clean

def _make_ocsvm():
    return SklearnDetector(
        _OCSVM(nu=0.1, kernel="rbf"),
        score_fn="decision_function")

def _make_iforest():
    return SklearnDetector(
        _IF(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1))

def _make_lof():
    return SklearnDetector(
        _LOF(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1))

def _make_gmm():
    return SklearnDetector(
        GaussianMixture(n_components=3, covariance_type="full",
                        random_state=42, max_iter=300))

def _make_ee():
    return SklearnDetector(
        _EE(contamination=0.05, support_fraction=0.9, random_state=42))


# ────────────────────────────────────────────────────────────────────────────
# PCA Null Subspace (unique logic — cannot be unified with SklearnDetector)
# ────────────────────────────────────────────────────────────────────────────

class PCANullSubspaceDetector:
    """
    PCA Null Subspace anomaly detector.

    Fit PCA on normal features; retain only the eigenvectors corresponding
    to the *lowest* variance directions (null subspace).

    Anomaly score = L2 norm of the projection of (x - mu) onto the null
    subspace:
        score(x) = || W_low^T (x - mu) ||_2

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

        pca_full = PCA().fit(Xs)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, self.retained_variance)) + 1
        self._W_low = pca_full.components_[k:].T   # (D, n_null)

        scores = self._score_raw(Xs)
        self._threshold = np.percentile(scores, 95)
        return self

    def _score_raw(self, Xs: np.ndarray) -> np.ndarray:
        centred = Xs - self._mean
        proj = centred @ self._W_low          # (N, n_null)
        return np.linalg.norm(proj, axis=1)   # (N,)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self._score_raw(self.scaler.transform(X))

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._threshold
        return (scores >= thr).astype(int)


# ────────────────────────────────────────────────────────────────────────────
# Supervised classifiers
# ────────────────────────────────────────────────────────────────────────────

class _SupervisedClassifier:
    """Wraps an sklearn classifier with PCA preprocessing."""

    def __init__(self, model, pca_variance: float = 0.95):
        self._pre = _PCAPreprocessor(pca_variance)
        self._model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._model.fit(Xp, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._pre.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._pre.transform(X))[:, 1]


# ────────────────────────────────────────────────────────────────────────────
# KDE Detector (Kernel Density Estimation)
# ────────────────────────────────────────────────────────────────────────────

class KDEDetector:
    """
    Kernel Density Estimation anomaly detector.
    Fit KDE on normal features; anomaly score = negative log-density.
    More flexible than GMM (no fixed number of components).
    Reference: scikit-learn KernelDensity.
    """
    def __init__(self, pca_variance: float = 0.95):
        from sklearn.neighbors import KernelDensity
        self._pre = _PCAPreprocessor(pca_variance)
        self._kde = KernelDensity(kernel="gaussian", bandwidth=1.0)
        self._threshold = None

    def fit(self, X: np.ndarray):
        Xp = self._pre.fit_transform(X)
        # Bandwidth selection via cross-validation
        from sklearn.model_selection import GridSearchCV
        bw_grid = {"bandwidth": [0.1, 0.5, 1.0, 2.0, 5.0]}
        grid = GridSearchCV(self._kde, bw_grid, cv=3)
        grid.fit(Xp)
        self._kde = grid.best_estimator_
        raw_scores = -self._kde.score_samples(Xp)  # neg log-density → higher = anomalous
        self._threshold = np.percentile(raw_scores, 90)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pre.transform(X)
        return -self._kde.score_samples(Xp)

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._threshold
        return (scores >= thr).astype(int)


# ────────────────────────────────────────────────────────────────────────────
# kNN Distance Detector
# ────────────────────────────────────────────────────────────────────────────

class KNNDetector:
    """
    k-Nearest Neighbor distance anomaly detector.
    Anomaly score = distance to k-th nearest neighbor in the normal training set.
    Reference: Nizan & Tal (2024), k-NNN.
    """
    def __init__(self, n_neighbors: int = 5, pca_variance: float = 0.95):
        from sklearn.neighbors import NearestNeighbors
        self._pre = _PCAPreprocessor(pca_variance)
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        self._k = n_neighbors
        self._threshold = None

    def fit(self, X: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._nn.fit(Xp)
        dists, _ = self._nn.kneighbors(Xp)
        raw_scores = dists[:, -1]  # distance to k-th neighbor
        self._threshold = np.percentile(raw_scores, 90)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pre.transform(X)
        dists, _ = self._nn.kneighbors(Xp)
        return dists[:, -1]

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._threshold
        return (scores >= thr).astype(int)


# ────────────────────────────────────────────────────────────────────────────
# Mahalanobis Distance Detector
# ────────────────────────────────────────────────────────────────────────────

class MahalanobisDetector:
    """
    Mahalanobis distance anomaly detector.
    Fit multivariate Gaussian on normal features using Ledoit-Wolf
    robust covariance estimator. Anomaly score = Mahalanobis distance.
    Classical analog of PaDiM (without CNN backbone).
    Reference: arXiv:2003.00402.
    """
    def __init__(self, pca_variance: float = 0.95):
        from sklearn.covariance import LedoitWolf
        self._pre = _PCAPreprocessor(pca_variance)
        self._cov = LedoitWolf()
        self._mean = None
        self._threshold = None

    def fit(self, X: np.ndarray):
        Xp = self._pre.fit_transform(X)
        self._mean = Xp.mean(axis=0)
        self._cov.fit(Xp)
        raw_scores = self._mahal(Xp)
        self._threshold = np.percentile(raw_scores, 90)
        return self

    def _mahal(self, X: np.ndarray) -> np.ndarray:
        diff = X - self._mean
        left = diff @ self._cov.precision_
        return np.sqrt(np.sum(left * diff, axis=1))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xp = self._pre.transform(X)
        return self._mahal(Xp)

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._threshold
        return (scores >= thr).astype(int)


# ────────────────────────────────────────────────────────────────────────────
# Ensemble Detector (score-level fusion)
# ────────────────────────────────────────────────────────────────────────────

class EnsembleDetector:
    """
    Ensemble anomaly detector: runs multiple base detectors, normalizes
    their scores to [0,1], and averages.
    """
    def __init__(self):
        self._detectors = [
            _make_iforest(),
            _make_ocsvm(),
            _make_lof(),
            _make_ee(),
        ]
        self._score_ranges = []
        self._threshold = None

    def fit(self, X: np.ndarray):
        self._score_ranges = []
        all_norm_scores = []
        for det in self._detectors:
            det.fit(X)
            scores = det.score_samples(X)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            self._score_ranges.append((p5, p95))
            norm = (scores - p5) / (p95 - p5 + 1e-9)
            all_norm_scores.append(np.clip(norm, 0, 1))
        avg_scores = np.mean(all_norm_scores, axis=0)
        self._threshold = np.percentile(avg_scores, 90)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        all_norm = []
        for det, (p5, p95) in zip(self._detectors, self._score_ranges):
            scores = det.score_samples(X)
            norm = (scores - p5) / (p95 - p5 + 1e-9)
            all_norm.append(np.clip(norm, 0, 1))
        return np.mean(all_norm, axis=0)

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._threshold
        return (scores >= thr).astype(int)


# ────────────────────────────────────────────────────────────────────────────
# Supervised classifiers
# ────────────────────────────────────────────────────────────────────────────

def _make_svm_clf():
    return _SupervisedClassifier(
        SVC(C=1.0, kernel="rbf", probability=True,
            class_weight="balanced", random_state=42))

def _make_rf_clf():
    return _SupervisedClassifier(
        _RF(n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1))


# ────────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────────

UNSUPERVISED_DETECTORS = {
    "OC-SVM":           _make_ocsvm,
    "IsolationForest":  _make_iforest,
    "LOF":              _make_lof,
    "GMM":              _make_gmm,
    "EllipticEnvelope": _make_ee,
    "PCANullSubspace":  PCANullSubspaceDetector,
    "KDE":              KDEDetector,
    "kNN":              KNNDetector,
    "Mahalanobis":      MahalanobisDetector,
    "Ensemble":         EnsembleDetector,
}

SUPERVISED_CLASSIFIERS = {
    "SVM":          _make_svm_clf,
    "RandomForest": _make_rf_clf,
}
