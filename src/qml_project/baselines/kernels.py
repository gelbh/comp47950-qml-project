"""Kernel helpers and diagnostics for classical Nim baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


def angle_encoding_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    M: int = 7,
) -> np.ndarray:
    r"""Kernel mimicking the angle-encoding quantum feature map (product state).

    Computes

    .. math::

        k(\mathbf x, \mathbf x')
        = \prod_{i=1}^{k} \cos^2\!\Bigl(\frac{(x_i - x'_i)\,\pi}{2}\Bigr)

    where :math:`x_i = h_i / M` are normalised heap sizes. This equals
    :math:`|\langle\psi(\mathbf x)|\psi(\mathbf x')\rangle|^2` for the
    product-state encoding
    :math:`|\psi(\mathbf x)\rangle = \bigotimes_i R_Y(h_i\pi/M)|0\rangle`.

    Parameters
    ----------
    X, Y : np.ndarray
        Feature arrays of normalised heap sizes, shapes ``(n, d)`` and
        ``(m, d)``.
    M : int
        Present for API consistency; the features must already be normalised
        by *M* (i.e. values in [0, 1]).

    Returns
    -------
    np.ndarray, shape ``(n, m)``
        Kernel (Gram) matrix.
    """
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (n, m, d)
    cos_sq = np.cos(diff * np.pi / 2) ** 2
    return np.prod(cos_sq, axis=2)


def _make_angle_kernel(M: int = 7):
    """Return a callable ``(X, Y) -> K`` suitable for ``SVC(kernel=...)``."""

    def _kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return angle_encoding_kernel(X, Y, M=M)

    return _kernel


def centered_kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """Compute centered kernel alignment between two Gram matrices.

    The value is in ``[-1, 1]`` when both kernels are centered:

    .. math::

        \mathrm{CKA}(K_1, K_2) =
        \frac{\langle K_{1,c}, K_{2,c}\rangle_F}
             {\|K_{1,c}\|_F\,\|K_{2,c}\|_F}

    where :math:`K_c = HKH` and :math:`H = I - \frac{1}{n}\mathbf 1\mathbf 1^T`.
    """
    if K1.shape != K2.shape:
        raise ValueError("K1 and K2 must have the same shape.")
    n = K1.shape[0]
    if n != K1.shape[1]:
        raise ValueError("Kernel matrices must be square.")

    H = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    K1c = H @ K1 @ H
    K2c = H @ K2 @ H
    num = float(np.sum(K1c * K2c))
    den = float(np.linalg.norm(K1c, ord="fro") * np.linalg.norm(K2c, ord="fro"))
    if den == 0.0:
        return 0.0
    return num / den


def label_kernel_binary(y: np.ndarray) -> np.ndarray:
    """Return binary target kernel ``yy^T`` using labels in {0, 1}."""
    y = np.asarray(y).ravel()
    labels = set(np.unique(y).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError("y must contain binary labels encoded as 0/1.")
    y_pm = 2.0 * y.astype(np.float64) - 1.0
    return np.outer(y_pm, y_pm)


def kernel_class_separation(
    K: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Summarise within-class vs between-class similarity in a kernel matrix."""
    if K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square Gram matrix.")
    y = np.asarray(y).ravel()
    if len(y) != K.shape[0]:
        raise ValueError("y length must match K size.")

    same = y[:, None] == y[None, :]
    diff = ~same
    diag = np.eye(len(y), dtype=bool)
    same_wo_diag = same & (~diag)

    same_vals = K[same_wo_diag]
    diff_vals = K[diff]
    mean_same = float(np.mean(same_vals)) if same_vals.size else 0.0
    mean_diff = float(np.mean(diff_vals)) if diff_vals.size else 0.0
    return {
        "mean_same_class": mean_same,
        "mean_diff_class": mean_diff,
        "gap_same_minus_diff": mean_same - mean_diff,
    }


def compare_kernels_for_nim(
    X: np.ndarray,
    y: np.ndarray,
    *,
    M: int = 7,
    rbf_gamma: float = 1.0,
) -> pd.DataFrame:
    """Compare angle and RBF kernels on Nim states.

    Returns a tidy DataFrame with quantitative diagnostics:
    centered alignment to the binary target kernel and same-vs-different class
    similarity gaps.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if len(y) != len(X):
        raise ValueError("y length must match number of rows in X.")

    K_angle = angle_encoding_kernel(X, X, M=M)
    K_rbf = rbf_kernel(X, X, gamma=rbf_gamma)
    K_target = label_kernel_binary(y)

    rows: list[dict[str, float | str]] = []
    kernels = {
        "angle": K_angle,
        "rbf": K_rbf,
    }
    for name, K in kernels.items():
        sep = kernel_class_separation(K, y)
        rows.append(
            {
                "kernel": name,
                "cka_to_target": centered_kernel_alignment(K, K_target),
                "mean_same_class": sep["mean_same_class"],
                "mean_diff_class": sep["mean_diff_class"],
                "gap_same_minus_diff": sep["gap_same_minus_diff"],
                "trace": float(np.trace(K)),
            }
        )

    return pd.DataFrame(rows)
