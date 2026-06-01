"""Variational classifier dataclass and circuit builder."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit

from .cz_pairs import AnsatzName, CZStrategy, cz_pairs_for_layer
from .decoding import bitstring_to_class_map


@dataclass
class VariationalClassifier:
    """A variational quantum classifier circuit and its metadata.

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
    ansatz : AnsatzName
        Per-qubit block style used in each layer.
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
    ansatz: AnsatzName
    _feat_idx: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _train_idx: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

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

    def bind(
        self,
        X: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """Build the parameter-values array expected by the Qiskit sampler.

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
    ansatz: AnsatzName = "basic_block",
) -> VariationalClassifier:
    """Construct a variational classifier circuit.

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
    ansatz : ``"basic_block"`` | ``"ry_rz"``
        Choice of the per-qubit ansatz block:

        - ``"basic_block"``: ``RX(pi/2) -> RZ(angle) -> RX(pi/2)``
        - ``"ry_rz"``: ``RY(angle) -> RZ(angle)``

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
    if ansatz not in ("basic_block", "ry_rz"):
        raise ValueError("ansatz must be 'basic_block' or 'ry_rz'")

    n_feat_layers_min = math.ceil(n_features / n_qubits)
    # Even-indexed layers are "feature"; we need enough of them to place every x[k].
    min_n_layers = 2 * n_feat_layers_min - 1
    if n_layers is not None and n_layers < min_n_layers:
        raise ValueError(
            f"n_layers={n_layers} is too small to bind all {n_features} feature "
            f"parameters across {n_qubits} qubits: need at least {min_n_layers} "
            "alternating block layers (feature layers on even indices). "
            "Increase n_layers, or reduce n_features / increase n_qubits."
        )
    if n_layers is None:
        n_layers = max(4, 2 * n_feat_layers_min + 2)

    # Even index → feature layer, odd → parameter layer.
    layer_types: list[str] = []
    for i in range(n_layers):
        layer_types.append("feature" if i % 2 == 0 else "param")

    # Feature layers consume features sequentially; once all features are used,
    # remaining slots become trainable parameters.
    feat_idx = 0
    n_trainable = 0
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

    feature_params = ParameterVector("x", n_features)
    trainable_params = ParameterVector("w", n_trainable)

    qc = QuantumCircuit(n_qubits)
    rng = np.random.default_rng(cz_seed)
    cz_pairs_per_layer: list[list[tuple[int, int]]] = []

    for layer_i in range(n_layers):
        for q in range(n_qubits):
            slot_type, slot_k = layer_slots[layer_i][q]
            angle = (
                feature_params[slot_k]
                if slot_type == "feature"
                else trainable_params[slot_k]
            )
            if ansatz == "basic_block":
                qc.rx(np.pi / 2, q)
                qc.rz(angle, q)
                qc.rx(np.pi / 2, q)
            else:
                qc.ry(angle, q)
                qc.rz(angle, q)

        if n_qubits >= 2:
            pairs = cz_pairs_for_layer(n_qubits, cz_strategy, rng)
            for q1, q2 in pairs:
                qc.cz(q1, q2)
            cz_pairs_per_layer.append(pairs)
        else:
            cz_pairs_per_layer.append([])

        if layer_i < n_layers - 1:
            qc.barrier()

    qc.measure_all()

    # ``circuit.parameters`` is sorted by name, so build an explicit name -> position
    # map and emit dense integer index arrays for fast bindings later.
    param_list = list(qc.parameters)
    param_to_pos = {p: i for i, p in enumerate(param_list)}
    feat_indices = np.array(
        [param_to_pos[feature_params[k]] for k in range(n_features)],
        dtype=np.intp,
    )
    train_indices = np.array(
        [param_to_pos[trainable_params[k]] for k in range(n_trainable)],
        dtype=np.intp,
    )

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
        ansatz=ansatz,
        _feat_idx=feat_indices,
        _train_idx=train_indices,
    )
