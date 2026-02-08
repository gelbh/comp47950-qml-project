"""
Preprocessing for COMP47950 QML project.

Implements centre/scale, optional PCA, and [20] Selig et al. angle mapping.
Fit on training data only; transform both train and test.
"""

from typing import NamedTuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# [20] Equation 2: f(x) = (1 - α²) π / q * W
# Paper defaults: απ = π/10 (so α = 0.1), q = 3
_ALPHA = 0.1
_QUANTILE = 3


def apply_angle_mapping(
    X: np.ndarray,
    *,
    alpha: float = _ALPHA,
    q: float = _QUANTILE,
) -> np.ndarray:
    """
    Apply [20] angle mapping: f(x) = (1 - α²) π / q * W.

    W is the input (expected to be centred/scaled). Paper defaults α=0.1, q=3.
    """
    scale = (1 - alpha**2) * np.pi / q
    return scale * X


class Preprocessor(NamedTuple):
    """Fitted scaler and optional PCA. Use fit_transform or transform."""

    scaler: StandardScaler
    pca: PCA | None

    def fit(self, X_train: np.ndarray) -> "Preprocessor":
        """Fit scaler (and PCA if configured) on training data. Returns self for chaining."""
        self.scaler.fit(X_train)
        if self.pca is not None:
            X_scaled = self.scaler.transform(X_train)
            self.pca.fit(X_scaled)
        return self

    def transform(
        self,
        X: np.ndarray,
        *,
        apply_angle_mapping_flag: bool = False,
        alpha: float = _ALPHA,
        q: float = _QUANTILE,
    ) -> np.ndarray:
        """Transform data: scale → (optional PCA) → (optional angle mapping)."""
        out = self.scaler.transform(X)
        if self.pca is not None:
            out = self.pca.transform(out)
        out = np.asarray(out)
        if apply_angle_mapping_flag:
            out = apply_angle_mapping(out, alpha=alpha, q=q)
        return out.astype(np.float64)

    def fit_transform(
        self,
        X_train: np.ndarray,
        *,
        apply_angle_mapping_flag: bool = False,
        alpha: float = _ALPHA,
        q: float = _QUANTILE,
    ) -> np.ndarray:
        """Fit on X_train and transform it."""
        self.fit(X_train)
        return self.transform(
            X_train,
            apply_angle_mapping_flag=apply_angle_mapping_flag,
            alpha=alpha,
            q=q,
        )


def make_preprocessor(n_components: int | None = None) -> Preprocessor:
    """Create a Preprocessor. Set n_components for PCA (Wine: 8 or 12, BC: 12 or 16)."""
    scaler = StandardScaler()
    pca = PCA(n_components=n_components) if n_components is not None else None
    return Preprocessor(scaler=scaler, pca=pca)


def preprocess(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_components: int | None = None,
    apply_angle_mapping_flag: bool = False,
) -> tuple[np.ndarray, np.ndarray, Preprocessor]:
    """
    Preprocess train and test: scale, optional PCA, optional [20] angle mapping.

    Returns:
        X_train_processed, X_test_processed, fitted Preprocessor
    """
    prep = make_preprocessor(n_components=n_components)
    X_train_out = prep.fit_transform(X_train, apply_angle_mapping_flag=apply_angle_mapping_flag)
    X_test_out = prep.transform(X_test, apply_angle_mapping_flag=apply_angle_mapping_flag)
    return X_train_out, X_test_out, prep
