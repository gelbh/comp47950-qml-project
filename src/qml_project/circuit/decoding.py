"""Bitstring-to-class mapping and decoding helpers for the VQC output layer."""

from __future__ import annotations

import numpy as np


def bitstring_to_class_map(n_qubits: int, n_classes: int) -> dict[int, int]:
    r"""Map bitstring integers to class labels by splitting into equal-size bins.

    The :math:`2^n` possible bitstrings are divided into *n_classes* contiguous
    ranges of size :math:`\lfloor 2^n / K \rfloor`. Leftover bitstrings (when
    :math:`2^n \mod K \neq 0`) are mapped to ``-1`` (ignored during evaluation).
    """
    n_bitstrings = 2**n_qubits
    bin_size = n_bitstrings // n_classes
    mapping: dict[int, int] = {}
    for bs in range(n_bitstrings):
        cls = bs // bin_size
        mapping[bs] = cls if cls < n_classes else -1
    return mapping


def counts_to_class_probs(
    counts: dict[str, int],
    n_qubits: int,
    n_classes: int,
    *,
    class_map: dict[int, int] | None = None,
) -> np.ndarray:
    """Convert measurement counts to class probabilities.

    Parameters
    ----------
    counts : dict[str, int]
        Measurement outcomes from the sampler, e.g. ``{"01": 50, "10": 30}``.
    n_qubits, n_classes : int
        Circuit and task dimensions.
    class_map : dict or None
        Pre-computed map from :func:`bitstring_to_class_map`. If *None*, it is
        built on the fly.

    Returns
    -------
    ndarray, shape ``(n_classes,)``
        Normalised class probabilities.
    """
    if class_map is None:
        class_map = bitstring_to_class_map(n_qubits, n_classes)

    class_counts = np.zeros(n_classes, dtype=np.float64)
    total_valid = 0

    for bitstring, count in counts.items():
        bs_int = int(bitstring, 2)
        cls = class_map.get(bs_int, -1)
        if cls >= 0:
            class_counts[cls] += count
            total_valid += count

    if total_valid == 0:
        return np.ones(n_classes, dtype=np.float64) / n_classes

    return class_counts / total_valid


def counts_to_z_expectation(
    counts: dict[str, int],
    *,
    qubit: int = 0,
) -> float:
    """Convert shot counts to expectation value ``<Z_qubit>``.

    Qiskit count keys are big-endian strings; qubit ``0`` corresponds to the
    right-most bit.
    """
    total = int(sum(counts.values()))
    if total == 0:
        return 0.0

    z_sum = 0.0
    for bitstring, count in counts.items():
        bit = bitstring[-1 - qubit]
        z_val = 1.0 if bit == "0" else -1.0
        z_sum += z_val * count
    return z_sum / total


def predict_from_probs(class_probs: np.ndarray) -> int:
    """Return the class with the highest probability."""
    return int(np.argmax(class_probs))


def predict_batch(class_probs_batch: np.ndarray) -> np.ndarray:
    """Return predicted class for each sample in a batch."""
    return np.argmax(class_probs_batch, axis=1)
