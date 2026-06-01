"""Stack per-state encoding helpers into VQC feature matrices."""

from __future__ import annotations

import numpy as np

from qml_project.nim.game import nim_sum

from .circuits import (
    SymmetryMode,
    _canonicalise_state,
    angle_parameters,
    binary_bits,
)


def angle_features_matrix(
    states: np.ndarray,
    *,
    M: int = 7,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Stack :func:`angle_parameters` per row for VQC feature layers (radians)."""
    arr = np.asarray(states, dtype=np.int32)
    rows: list[np.ndarray] = []
    for row in arr:
        tup = tuple(int(x) for x in row.tolist())
        rows.append(
            angle_parameters(
                tup, M=M, include_nim_sum=include_nim_sum, symmetry=symmetry
            )
        )
    return np.asarray(rows, dtype=np.float64)


def binary_angle_features_matrix(
    states: np.ndarray,
    *,
    bits_per_heap: int = 3,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Heap and Nim-sum-register bits as ``0`` or ``π`` (VQC ``RY`` angles).

    Row width ``4 * bits_per_heap`` matches :func:`build_binary_encoding_circuit`.
    When ``include_nim_sum`` is false, the Nim-sum register features are zero.
    """
    arr = np.asarray(states, dtype=np.int32)
    B = bits_per_heap
    out = np.zeros((arr.shape[0], 4 * B), dtype=np.float64)
    for i, row in enumerate(arr):
        tup = tuple(int(x) for x in row.tolist())
        heap = binary_bits(tup, bits_per_heap=B, symmetry=symmetry).astype(np.float64)
        out[i, : 3 * B] = heap * np.pi
        state_sym = _canonicalise_state(tup, symmetry=symmetry)
        ns = int(nim_sum(state_sym))
        if ns < 0:
            raise ValueError("nim_sum must be non-negative")
        if ns >= 2**B:
            raise ValueError(
                f"nim_sum={ns} does not fit in bits_per_heap={B} "
                "(increase bits_per_heap or reduce heap range)"
            )
        tail = np.zeros(B, dtype=np.float64)
        if include_nim_sum:
            for bit in range(B):
                tail[bit] = float((ns >> bit) & 1)
        out[i, 3 * B :] = tail * np.pi
    return out
