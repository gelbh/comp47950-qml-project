"""VQC device payload, refit, pub construction, and counts decoding.

Section 10 submits the VQC winner to IBM Runtime. The VQC inference cost
is train-size-independent, but we still refit at the device train budget
(``DEVICE_TRAIN_SIZE``) so the device comparison is apples-to-apples with
QSVM, whose support-vector count grows with the training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from qml_project.circuit import (
    AnsatzName,
    CZStrategy,
    VariationalClassifier,
    build_circuit,
    counts_to_class_probs,
    predict_from_probs,
)
from qml_project.nim.encoding import EncodingName, SymmetryMode, amplitude_vector
from qml_project.nim.state_utils import state_tuple_from_array
from qml_project.training.types import (
    DecisionRule,
    LossName,
    MeasurementObservable,
)
from qml_project.vqc_workflow import transform_states_for_vqc, vqc_encoding_profile


def _amplitude_features_for_vqc(
    states: np.ndarray,
    *,
    M: int = 7,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Replicate Section 05's amplitude feature transform for VQC inputs.

    The VQC winner uses ``encoding='amplitude'``: each state is mapped to
    an amplitude vector (length ``2^n_qubits``) and then multiplied by
    ``π`` to get angle-like features. Keeping this local avoids Section 10
    having to depend on the notebook-level helper in ``05_vqc_workflow``.
    """
    rows = [
        amplitude_vector(
            state_tuple_from_array(s),
            M=M,
            include_nim_sum=include_nim_sum,
            symmetry=symmetry,
        )
        for s in np.asarray(states, dtype=np.int32)
    ]
    return (np.asarray(rows, dtype=np.float64) * np.pi).astype(np.float64)


@dataclass
class VQCDevicePayload:
    """Serialisable VQC artefact ready for device submission.

    Holds the built circuit, trained parameters, and the metadata needed
    to (a) reproduce the feature transform on raw Nim states at inference
    time, and (b) decode per-sample counts back into class predictions.
    """

    config_id: str
    encoding: EncodingName
    ansatz: AnsatzName
    n_qubits: int
    n_features: int
    n_classes: int
    n_layers: int
    cz_strategy: CZStrategy
    cz_seed: int
    decision_rule: DecisionRule
    observable: MeasurementObservable
    loss_name: LossName
    symmetry: SymmetryMode
    theta: np.ndarray
    circuit: QuantumCircuit
    feature_kwargs: Mapping[str, Any] = field(default_factory=dict)
    train_size_used: int | None = None
    refit_seed: int = 0
    refit_balanced_accuracy: float | None = None

    def build_vqc(self) -> VariationalClassifier:
        """Rebuild the ``VariationalClassifier`` wrapper from the stored config."""
        return build_circuit(
            n_qubits=self.n_qubits,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_layers=self.n_layers,
            cz_strategy=self.cz_strategy,
            cz_seed=self.cz_seed,
            ansatz=self.ansatz,
        )

    def feature_transform(self, states: np.ndarray) -> np.ndarray:
        """Reproduce the training-time feature transform on raw Nim states."""
        fk = dict(self.feature_kwargs)
        if self.encoding == "amplitude":
            return _amplitude_features_for_vqc(states, **fk)
        if self.encoding == "angle":
            return transform_states_for_vqc(
                states,
                encoding="angle",
                M=int(fk.get("M", 7)),
                include_nim_sum=bool(fk.get("include_nim_sum", True)),
            )
        if self.encoding == "binary":
            return transform_states_for_vqc(
                states,
                encoding="binary",
                M=int(fk.get("M", 7)),
                bits_per_heap=int(fk.get("bits_per_heap", 3)),
                include_nim_sum=bool(fk.get("include_nim_sum", True)),
            )
        raise NotImplementedError(
            f"device inference supports encoding='amplitude', 'angle', or 'binary'; "
            f"winner uses {self.encoding!r}."
        )


def refit_vqc_for_device(
    *,
    winner_row: Mapping[str, Any],
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    train_size: int = 50,
    max_iter: int = 200,
    seed: int = 0,
    test_shots: int = 300,
) -> VQCDevicePayload:
    """Refit a VQC at the winner's hyperparameters for device submission.

    ``winner_row`` is one row from the per-seed VQC workflow frame — it
    carries the full hyperparameter set (``ansatz``, ``n_layers``,
    ``cz_strategy``, ``encoding``, ``decision_rule``, ``observable``,
    ``loss_name``). ``train_size`` is the device-refit budget (default 50;
    see Section 08).
    """
    from sklearn.metrics import balanced_accuracy_score

    from qml_project.nim.data import training_subsets
    from qml_project.training.evaluation import (
        evaluate_circuit_outputs,
        train_classifier,
    )

    encoding = str(winner_row["encoding"])
    ansatz = str(winner_row["ansatz"])
    n_layers = int(winner_row["n_layers"])
    cz_strategy = str(winner_row["cz_strategy"])
    decision_rule = str(winner_row.get("decision_rule", "argmax"))
    observable = str(winner_row.get("observable", "bitstring_probs"))
    loss_name = str(winner_row.get("loss_name", "softmax_nll"))
    symmetry = str(winner_row.get("symmetry", "none"))
    include_nim_sum = bool(winner_row.get("include_nim_sum", True))
    bits_per_heap = int(winner_row.get("bits_per_heap", 3))

    if encoding == "amplitude":
        vec_sample = amplitude_vector(
            state_tuple_from_array(np.asarray(X_train_raw)[0]),
            M=7,
            include_nim_sum=include_nim_sum,
            symmetry=symmetry if symmetry in ("none", "canonical") else "none",
        )
        n_features = int(vec_sample.size)
        n_qubits = int(np.log2(n_features))
        if 2**n_qubits != n_features:
            raise ValueError("amplitude vector length must be a power of two")

        X_train_feat = _amplitude_features_for_vqc(
            X_train_raw,
            M=7,
            include_nim_sum=include_nim_sum,
            symmetry="none",
        )
        feature_kwargs: dict[str, Any] = {
            "M": 7,
            "include_nim_sum": include_nim_sum,
            "symmetry": "none",
        }
    elif encoding in ("angle", "binary"):
        prof = vqc_encoding_profile(
            encoding,
            include_nim_sum=include_nim_sum,
            bits_per_heap=bits_per_heap,
        )
        n_qubits = int(prof["n_qubits"])
        n_features = int(prof["n_features"])
        X_train_feat = transform_states_for_vqc(
            X_train_raw,
            encoding=encoding,
            M=7,
            bits_per_heap=bits_per_heap,
            include_nim_sum=include_nim_sum,
        )
        feature_kwargs = {
            "M": 7,
            "include_nim_sum": include_nim_sum,
            "bits_per_heap": bits_per_heap,
        }
    else:
        raise NotImplementedError(
            f"device refit supports encoding='amplitude', 'angle', or 'binary'; "
            f"winner uses encoding={encoding!r}."
        )
    # ``training_subsets`` drops any size ``>= len(X_train)`` and only stores
    # the full set under the ``"full"`` key. Clamp and route to ``"full"``
    # when the requested device-refit size meets/exceeds the training set
    # so we still get a valid :class:`TrainSubset`.
    _n_train_full = int(len(X_train_feat))
    _effective_size = int(train_size)
    if _effective_size >= _n_train_full:
        _effective_size = _n_train_full
        subsets = training_subsets(
            X_train_feat,
            np.asarray(y_train),
            sizes=[],
            random_state=int(seed),
        )
        subset = subsets["full"]
    else:
        subsets = training_subsets(
            X_train_feat,
            np.asarray(y_train),
            sizes=[_effective_size],
            random_state=int(seed),
        )
        subset = subsets[_effective_size]

    vc = build_circuit(
        n_qubits=n_qubits,
        n_features=n_features,
        n_classes=2,
        n_layers=n_layers,
        cz_strategy=cz_strategy,  # type: ignore[arg-type]
        cz_seed=42,
        ansatz=ansatz,  # type: ignore[arg-type]
    )
    theta, _history = train_classifier(
        vc,
        subset.X,
        subset.y,
        None,
        None,
        max_iter=int(max_iter),
        seed=int(seed),
        test_shots=int(test_shots),
        decision_rule=decision_rule,  # type: ignore[arg-type]
        observable=observable,  # type: ignore[arg-type]
        loss_name=loss_name,  # type: ignore[arg-type]
        verbose=False,
    )

    refit_accuracy: float | None = None
    if X_test_raw is not None and y_test is not None:
        from qiskit.primitives import StatevectorSampler

        sampler = StatevectorSampler(seed=int(seed))
        if encoding == "amplitude":
            X_test_feat = _amplitude_features_for_vqc(
                X_test_raw,
                M=7,
                include_nim_sum=include_nim_sum,
                symmetry="none",
            )
        else:
            X_test_feat = transform_states_for_vqc(
                X_test_raw,
                encoding=encoding,
                M=7,
                bits_per_heap=bits_per_heap,
                include_nim_sum=include_nim_sum,
            )
        outputs = evaluate_circuit_outputs(
            vc, X_test_feat, theta, int(test_shots), sampler
        )
        preds = np.argmax(outputs["class_probs"], axis=1)
        refit_accuracy = float(balanced_accuracy_score(np.asarray(y_test), preds))

    return VQCDevicePayload(
        config_id=str(winner_row["config_id"]),
        encoding=encoding,  # type: ignore[arg-type]
        ansatz=ansatz,  # type: ignore[arg-type]
        n_qubits=int(n_qubits),
        n_features=int(n_features),
        n_classes=2,
        n_layers=int(n_layers),
        cz_strategy=cz_strategy,  # type: ignore[arg-type]
        cz_seed=42,
        decision_rule=decision_rule,  # type: ignore[arg-type]
        observable=observable,  # type: ignore[arg-type]
        loss_name=loss_name,  # type: ignore[arg-type]
        symmetry=symmetry,  # type: ignore[arg-type]
        theta=np.asarray(theta, dtype=np.float64),
        circuit=vc.circuit,
        feature_kwargs=feature_kwargs,
        train_size_used=int(_effective_size),
        refit_seed=int(seed),
        refit_balanced_accuracy=refit_accuracy,
    )


def build_vqc_device_pubs(
    payload: VQCDevicePayload,
    X_test_raw: np.ndarray,
    *,
    shots: int = 1024,
) -> list[SamplerPub]:
    """Build a single ``SamplerPub`` batching all test samples for a VQC run.

    One pub with a ``(n_samples, n_params)`` bindings array is cheaper
    than one pub per sample: the SamplerV2 primitive batches samples at
    the backend level and we avoid N job submissions on queue-limited
    free-tier Runtime.
    """
    vc = payload.build_vqc()
    X_feat = payload.feature_transform(np.asarray(X_test_raw))
    bound_values = vc.bind(X_feat, payload.theta)
    bindings = BindingsArray({tuple(vc.circuit.parameters): bound_values})
    pub = SamplerPub(circuit=vc.circuit, parameter_values=bindings, shots=int(shots))
    return [pub]


def decode_vqc_counts(
    counts_list: Sequence[Mapping[str, int]],
    payload: VQCDevicePayload,
) -> np.ndarray:
    """Decode per-sample counts to class predictions using the winner's rule."""
    vc = payload.build_vqc()
    preds = np.empty(len(counts_list), dtype=np.int64)
    for i, counts in enumerate(counts_list):
        class_probs = counts_to_class_probs(
            dict(counts),
            n_qubits=payload.n_qubits,
            n_classes=payload.n_classes,
            class_map=vc.class_map,
        )
        if payload.decision_rule == "argmax":
            preds[i] = predict_from_probs(class_probs)
        else:
            p0 = float(class_probs[0])
            preds[i] = int(p0 < 0.5)
    return preds


__all__ = [
    "VQCDevicePayload",
    "build_vqc_device_pubs",
    "decode_vqc_counts",
    "refit_vqc_for_device",
]
