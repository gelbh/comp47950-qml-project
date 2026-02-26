"""
Variational quantum classifier.

Implements the circuit design:
  - Basic block: RX(π/2) → RZ(·) → RX(π/2), one per qubit per layer.
  - Alternating feature and parameter layers with CZ entanglement.
  - Bitstring-to-class output mapping (equal-range bins).
  - Softmax negative-log-likelihood loss.

Designed for Qiskit ≥ 2.0 (QuantumCircuit, ParameterVector, StatevectorSampler).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

CZStrategy = Literal["linear", "all", "random"]


# ---------------------------------------------------------------------------
# CZ-pair generation
# ---------------------------------------------------------------------------


def _all_qubit_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """All unique qubit pairs (i, j) with i < j."""
    return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]


def cz_pairs_for_layer(
    n_qubits: int,
    strategy: CZStrategy,
    rng: np.random.Generator,
    max_pairs: int = 3,
) -> list[tuple[int, int]]:
    """
    Select CZ-gate qubit pairs for one layer.

    Strategies:
      - ``"linear"``: Adjacent pairs (0,1), (1,2), …
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


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------


@dataclass
class VariationalClassifier:
    """
    A variational quantum classifier circuit and its metadata.

    Attributes
    ----------
    circuit : QuantumCircuit
        The parameterized Qiskit circuit (includes measurement).
    feature_params : ParameterVector
        Feature parameters (named ``x[0]``, ``x[1]``, …).
    trainable_params : ParameterVector
        Trainable weight parameters (named ``w[0]``, ``w[1]``, …).
    n_qubits, n_features, n_classes : int
        Circuit dimensions.
    class_map : dict[int, int]
        Mapping from bitstring integer to class label (-1 = unassigned).
    cz_pairs_per_layer : list[list[tuple[int, int]]]
        CZ pairs used in each layer (for inspection / reproducibility).
    layer_types : list[str]
        ``"feature"`` or ``"param"`` for each layer.
    """

    circuit: QuantumCircuit
    feature_params: ParameterVector
    trainable_params: ParameterVector
    n_qubits: int
    n_features: int
    n_classes: int
    class_map: dict[int, int]
    cz_pairs_per_layer: list[list[tuple[int, int]]]
    layer_types: list[str]
    # Private: indices into circuit.parameters for fast binding
    _feat_idx: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _train_idx: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    # -- properties ----------------------------------------------------------

    @property
    def n_trainable(self) -> int:
        return len(self.trainable_params)

    @property
    def n_layers(self) -> int:
        return len(self.layer_types)

    @property
    def n_params(self) -> int:
        """Total number of parameters (features + trainable)."""
        return self.n_features + self.n_trainable

    # -- parameter binding ---------------------------------------------------

    def bind(
        self,
        X: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Build the parameter-values array expected by the Qiskit sampler.

        Parameters
        ----------
        X : ndarray, shape ``(n_samples, n_features)`` or ``(n_features,)``
            Feature values (angle-mapped).
        theta : ndarray, shape ``(n_trainable,)``
            Current trainable weights.

        Returns
        -------
        ndarray, shape ``(n_samples, n_params)``
            Ready to pass as the second element of a ``SamplerPub``.
        """
        if X.ndim == 1:
            X = X[np.newaxis, :]
        n_samples = X.shape[0]
        values = np.empty((n_samples, self.n_params), dtype=np.float64)
        values[:, self._feat_idx] = X
        values[:, self._train_idx] = theta[np.newaxis, :]
        return values


def build_circuit(
    n_qubits: int,
    n_features: int,
    n_classes: int,
    *,
    n_layers: int | None = None,
    cz_strategy: CZStrategy = "linear",
    cz_seed: int = 42,
) -> VariationalClassifier:
    """
    Construct a variational classifier circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (should be ≤ *n_features*).
    n_features : int
        Number of input features (after preprocessing / angle mapping).
    n_classes : int
        Number of target classes.
    n_layers : int or None
        Total number of block layers (alternating feature / parameter).
        Default: ``2 * ceil(n_features / n_qubits) + 2`` (ensures ≥ 1 feature
        layer covering all features, plus parameter layers for expressivity).
    cz_strategy : ``"linear"`` | ``"all"`` | ``"random"``
        CZ-gate pair selection strategy.
    cz_seed : int
        RNG seed for ``"random"`` CZ strategy (ignored otherwise).

    Returns
    -------
    VariationalClassifier
        Dataclass containing the circuit, parameters, and metadata.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be ≥ 1")
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")
    if n_classes < 2:
        raise ValueError("n_classes must be ≥ 2")

    # Minimum feature layers to encode every feature at least once
    n_feat_layers_min = math.ceil(n_features / n_qubits)
    if n_layers is None:
        # Feature-param alternation: ensure all features + some trainability
        n_layers = max(4, 2 * n_feat_layers_min + 2)

    # Assign layer types: even index → feature, odd → parameter
    layer_types: list[str] = []
    for i in range(n_layers):
        layer_types.append("feature" if i % 2 == 0 else "param")

    # Assign per-qubit slots.  Feature layers consume features sequentially;
    # once all features are used, remaining slots become trainable parameters.
    feat_idx = 0
    n_trainable = 0
    # layer_slots[layer][qubit] = ("feature", k) or ("param", k)
    layer_slots: list[list[tuple[str, int]]] = []

    for layer_i in range(n_layers):
        qubit_slots: list[tuple[str, int]] = []
        for _q in range(n_qubits):
            if layer_types[layer_i] == "feature" and feat_idx < n_features:
                qubit_slots.append(("feature", feat_idx))
                feat_idx += 1
            else:
                qubit_slots.append(("param", n_trainable))
                n_trainable += 1
        layer_slots.append(qubit_slots)

    # Create parameter vectors
    feature_params = ParameterVector("x", n_features)
    trainable_params = ParameterVector("w", n_trainable)

    # Build quantum circuit
    qc = QuantumCircuit(n_qubits)
    rng = np.random.default_rng(cz_seed)
    cz_pairs_per_layer: list[list[tuple[int, int]]] = []

    for layer_i in range(n_layers):
        # Basic block per qubit: RX(π/2) → RZ(angle) → RX(π/2)
        for q in range(n_qubits):
            slot_type, slot_k = layer_slots[layer_i][q]
            angle = (
                feature_params[slot_k]
                if slot_type == "feature"
                else trainable_params[slot_k]
            )
            qc.rx(np.pi / 2, q)
            qc.rz(angle, q)
            qc.rx(np.pi / 2, q)

        # CZ entanglement
        if n_qubits >= 2:
            pairs = cz_pairs_for_layer(n_qubits, cz_strategy, rng)
            for q1, q2 in pairs:
                qc.cz(q1, q2)
            cz_pairs_per_layer.append(pairs)
        else:
            cz_pairs_per_layer.append([])

        # Visual barrier between layers
        if layer_i < n_layers - 1:
            qc.barrier()

    # Measurement in Z-basis on all qubits
    qc.measure_all()

    # Build parameter-index mapping (circuit.parameters is sorted by name)
    param_list = list(qc.parameters)
    param_to_pos = {p: i for i, p in enumerate(param_list)}
    feat_indices = np.array(
        [param_to_pos[feature_params[k]] for k in range(n_features)]
    )
    train_indices = np.array(
        [param_to_pos[trainable_params[k]] for k in range(n_trainable)]
    )

    # Bitstring-to-class mapping
    class_map = bitstring_to_class_map(n_qubits, n_classes)

    return VariationalClassifier(
        circuit=qc,
        feature_params=feature_params,
        trainable_params=trainable_params,
        n_qubits=n_qubits,
        n_features=n_features,
        n_classes=n_classes,
        class_map=class_map,
        cz_pairs_per_layer=cz_pairs_per_layer,
        layer_types=layer_types,
        _feat_idx=feat_indices,
        _train_idx=train_indices,
    )


# ---------------------------------------------------------------------------
# Output mapping
# ---------------------------------------------------------------------------


def bitstring_to_class_map(n_qubits: int, n_classes: int) -> dict[int, int]:
    """
    Map bitstring integers → class labels by splitting into equal-size bins.

    The $2^n$ possible bitstrings are divided into *n_classes* contiguous
    ranges of size $\\lfloor 2^n / K \\rfloor$.  Leftover bitstrings (when
    $2^n \\mod K \\neq 0$) are mapped to ``-1`` (ignored during evaluation).
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
    """
    Convert measurement counts to class probabilities.

    Parameters
    ----------
    counts : dict[str, int]
        Measurement outcomes from the sampler, e.g. ``{"01": 50, "10": 30}``.
    n_qubits, n_classes : int
        Circuit / task dimensions.
    class_map : dict or None
        Pre-computed map (from :func:`bitstring_to_class_map`).  If *None*,
        it is built on the fly.

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


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def softmax_nll_loss(
    class_probs: np.ndarray,
    true_label: int,
    *,
    eps: float = 1e-10,
) -> float:
    """
    Softmax negative-log-likelihood loss.

    $$\\mathcal{L}(x, y) = -\\log\\frac{e^{P_y}}{\\sum_{k} e^{P_k}}$$

    where $P_k$ are the class probabilities from measurement.
    """
    exp_p = np.exp(class_probs - np.max(class_probs))  # numerically stable
    softmax_p = exp_p / (exp_p.sum() + eps)
    return float(-np.log(softmax_p[true_label] + eps))


def batch_loss(
    class_probs_batch: np.ndarray,
    true_labels: np.ndarray,
    *,
    eps: float = 1e-10,
) -> float:
    """
    Mean softmax NLL loss over a batch.

    Parameters
    ----------
    class_probs_batch : ndarray, shape ``(batch_size, n_classes)``
    true_labels : ndarray, shape ``(batch_size,)``
    """
    n = len(true_labels)
    total = 0.0
    for i in range(n):
        total += softmax_nll_loss(class_probs_batch[i], int(true_labels[i]), eps=eps)
    return total / n


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------


def predict_from_probs(class_probs: np.ndarray) -> int:
    """Return the class with the highest probability."""
    return int(np.argmax(class_probs))


def predict_batch(class_probs_batch: np.ndarray) -> np.ndarray:
    """Return predicted class for each sample in a batch."""
    return np.argmax(class_probs_batch, axis=1)
