"""Feature engineering for classical Nim baselines."""

from __future__ import annotations

from typing import Literal

import numpy as np

from qml_project.nim.data import normalise_states

FeatureSet = Literal["raw", "parity", "heap_parity", "pairwise_xor", "bit_parity"]

FEATURE_SET_DESCRIPTIONS: dict[str, str] = {
    "raw": "Normalised heaps (3)",
    "heap_parity": "+ heap parities (6)",
    "pairwise_xor": "+ pairwise XOR (6)",
    "bit_parity": "+ column bit parities (6)",
    "parity": "All parity features (12)",
}

# Parity-style subsets for the §3.4 SVM ablation (raw baseline lives in the main sweep).
PARITY_ABLATION_FEATURE_SETS: tuple[str, ...] = (
    "heap_parity",
    "pairwise_xor",
    "bit_parity",
    "parity",
)


def _heap_parities(states: np.ndarray) -> np.ndarray:
    """Per-heap parities: ``h_i mod 2``."""
    return (states % 2).astype(np.float64)


def _pairwise_xor(states: np.ndarray, M: int) -> np.ndarray:
    """Pairwise XOR of heap sizes, normalised by *M*."""
    n, k = states.shape
    if k < 2:
        return np.empty((n, 0), dtype=np.float64)
    i_idx, j_idx = np.triu_indices(k, k=1)
    xored = (states[:, i_idx] ^ states[:, j_idx]).astype(np.float64) / M
    return xored


def _bit_parities(states: np.ndarray, M: int) -> np.ndarray:
    """Column-wise bit parities (individual bits of the Nim-sum)."""
    n, _k = states.shape
    n_bits = int(np.ceil(np.log2(M + 1)))
    if n_bits == 0:
        return np.empty((n, 0), dtype=np.float64)
    s = states.astype(np.int32, copy=False)
    bit_axes = np.arange(n_bits, dtype=np.int32)
    bits = (s[:, :, np.newaxis] >> bit_axes) & 1
    return np.bitwise_xor.reduce(bits, axis=1).astype(np.float64)


def engineer_parity_features(states: np.ndarray, *, M: int = 7) -> np.ndarray:
    """Add parity / XOR features to raw heap-size arrays.

    Appends to each row:
      - Heap parities: ``h_i mod 2`` for each heap  (k features)
      - Pairwise XOR:  ``h_i ⊕ h_j`` for all pairs  (k*(k-1)/2 features)
      - Column-wise bit parities: XOR of bit *b* across all heaps
        for each bit position (``ceil(log2(M+1))`` features)

    Parameters
    ----------
    states : np.ndarray, shape ``(n, k)``
        Raw (unnormalised) integer heap sizes.
    M : int
        Maximum heap size (determines number of bit columns).

    Returns
    -------
    np.ndarray, shape ``(n, k + k + k*(k-1)/2 + n_bits)``
        Normalised heaps concatenated with engineered features.
    """
    norm = normalise_states(states, M_max=M)
    return np.hstack(
        [
            norm,
            _heap_parities(states),
            _pairwise_xor(states, M),
            _bit_parities(states, M),
        ]
    )


def prepare_features(
    states: np.ndarray,
    feature_set: FeatureSet = "raw",
    *,
    M: int = 7,
) -> np.ndarray:
    """Transform raw heap sizes into features for a given feature set.

    Parameters
    ----------
    states : np.ndarray, shape ``(n, k)``
        Raw integer heap sizes.
    feature_set : FeatureSet
        ``"raw"``          — normalised heap sizes only (3 features).
        ``"heap_parity"``  — raw + per-heap parities (6 features).
        ``"pairwise_xor"`` — raw + pairwise XOR (6 features).
        ``"bit_parity"``   — raw + column-wise bit parities (6 features).
        ``"parity"``       — all of the above (12 features).
    M : int
        Maximum heap size for normalisation.
    """
    if feature_set == "raw":
        return normalise_states(states, M_max=M)
    if feature_set == "parity":
        return engineer_parity_features(states, M=M)
    norm = normalise_states(states, M_max=M)
    if feature_set == "heap_parity":
        return np.hstack([norm, _heap_parities(states)])
    if feature_set == "pairwise_xor":
        return np.hstack([norm, _pairwise_xor(states, M)])
    if feature_set == "bit_parity":
        return np.hstack([norm, _bit_parities(states, M)])
    raise ValueError(f"Unknown feature_set: {feature_set!r}")
