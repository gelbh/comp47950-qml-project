"""Shared types and CZ-pair generation for the variational classifier ansatz."""

from __future__ import annotations

from typing import Literal

import numpy as np

CZStrategy = Literal["linear", "all", "random"]
AnsatzName = Literal["basic_block", "ry_rz"]


def _all_qubit_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """All unique qubit pairs (i, j) with i < j."""
    return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]


def cz_pairs_for_layer(
    n_qubits: int,
    strategy: CZStrategy,
    rng: np.random.Generator,
    max_pairs: int = 3,
) -> list[tuple[int, int]]:
    """Select CZ-gate qubit pairs for one layer.

    Strategies:
      - ``"linear"``: Adjacent pairs ``(0, 1)``, ``(1, 2)``, …
      - ``"all"``: Every unique pair.
      - ``"random"``: 1 to *max_pairs* randomly chosen unique pairs.
    """
    if n_qubits < 2:
        return []

    if strategy == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]

    all_pairs = _all_qubit_pairs(n_qubits)

    if strategy == "random":
        n_choose = int(rng.integers(1, min(max_pairs, len(all_pairs)) + 1))
        indices = rng.choice(len(all_pairs), size=n_choose, replace=False)
        return [all_pairs[int(i)] for i in sorted(indices)]

    return all_pairs
