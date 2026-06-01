"""Shot schedule, circuit execution, and bitstring decoding for the VQC."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from qml_project.circuit import VariationalClassifier, batch_loss, predict_batch
from qml_project.training.noise_aer import mitigate_readout_prob_vector
from qml_project.training.types import DecisionRule, LossName

DEFAULT_SHOT_SCHEDULE: dict[int, int] = {1: 250, 21: 500, 51: 750}


def shots_for_eval(
    eval_number: int,
    schedule: dict[int, int] | None = None,
) -> int:
    """Return the shot count for a given function-evaluation number.

    Default schedule:
      - Evaluations 1-20:  250 shots
      - Evaluations 21-50: 500 shots
      - Evaluations 51+:   750 shots
    """
    if schedule is None:
        schedule = DEFAULT_SHOT_SCHEDULE
    qualifying = [t for t in sorted(schedule) if eval_number >= t]
    return schedule[qualifying[-1]] if qualifying else 250


def evaluate_circuit(
    vc: VariationalClassifier,
    X: np.ndarray,
    theta: np.ndarray,
    shots: int,
    sampler: Any,
) -> np.ndarray:
    """Run the parameterised circuit on all samples and return class probabilities.

    Parameters
    ----------
    vc : VariationalClassifier
        The circuit (with feature + trainable parameter slots).
    X : ndarray, shape ``(n_samples, n_features)``
        Angle-mapped input features.
    theta : ndarray, shape ``(n_trainable,)``
        Current trainable weights.
    shots : int
        Number of measurement shots per sample.
    sampler
        A Qiskit V2 sampler (``StatevectorSampler`` or Aer ``SamplerV2``).

    Returns
    -------
    ndarray, shape ``(n_samples, n_classes)``
        Class probability matrix.
    """
    outputs = evaluate_circuit_outputs(vc, X, theta, shots, sampler)
    return outputs["class_probs"]


def evaluate_circuit_outputs(
    vc: VariationalClassifier,
    X: np.ndarray,
    theta: np.ndarray,
    shots: int,
    sampler: Any,
    *,
    expectation_qubit: int = 0,
    readout_assignment_matrix: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the circuit and return class probabilities plus ``<Z>`` expectations."""
    n_samples = X.shape[0]
    bound_values = vc.bind(X, theta)

    ba = BindingsArray({tuple(vc.circuit.parameters): bound_values})
    pub = SamplerPub(circuit=vc.circuit, parameter_values=ba, shots=shots)
    job = sampler.run([pub])
    result = job.result()

    class_probs = np.zeros((n_samples, vc.n_classes), dtype=np.float64)
    z_expectations = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        counts = result[0].data.meas.get_counts(i)
        probs = _counts_to_prob_vector(counts, vc.n_qubits)
        if readout_assignment_matrix is not None:
            probs = mitigate_readout_prob_vector(probs, readout_assignment_matrix)
        class_probs[i] = _class_probs_from_bitstring_probs(
            probs, vc.n_classes, vc.class_map
        )
        z_expectations[i] = _z_expectation_from_bitstring_probs(
            probs, qubit=expectation_qubit
        )

    return {"class_probs": class_probs, "z_expectations": z_expectations}


def _expectation_to_p1(z_expectations: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Map ``<Z>`` in ``[-1, 1]`` to ``p(class=1)`` in ``[0, 1]``."""
    p1 = 0.5 * (1.0 - z_expectations)
    return np.clip(p1, eps, 1.0 - eps)


def _counts_to_prob_vector(
    counts: dict[str, int],
    n_qubits: int,
) -> np.ndarray:
    """Convert counts dictionary to bitstring probability vector."""
    n_states = 2**n_qubits
    probs = np.zeros(n_states, dtype=np.float64)
    total = float(sum(counts.values()))
    if total <= 0:
        return np.ones(n_states, dtype=np.float64) / float(n_states)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        probs[idx] += float(count)
    return probs / total


def _class_probs_from_bitstring_probs(
    bitstring_probs: np.ndarray,
    n_classes: int,
    class_map: dict[int, int],
) -> np.ndarray:
    """Aggregate bitstring probabilities into class probabilities."""
    out = np.zeros(n_classes, dtype=np.float64)
    total = 0.0
    for idx, p in enumerate(bitstring_probs):
        cls = class_map.get(int(idx), -1)
        if cls >= 0:
            out[cls] += float(p)
            total += float(p)
    if total <= 1e-15:
        return np.ones(n_classes, dtype=np.float64) / float(n_classes)
    return out / total


def _z_expectation_from_bitstring_probs(
    bitstring_probs: np.ndarray,
    *,
    qubit: int = 0,
) -> float:
    """Compute ``<Z_qubit>`` from full bitstring probabilities."""
    n_states = int(bitstring_probs.shape[0])
    n_qubits = int(np.log2(max(1, n_states)))
    if n_states == 0:
        return 0.0
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(f"qubit must be in [0, {n_qubits - 1}]")
    z = 0.0
    for idx, p in enumerate(bitstring_probs):
        bit = (idx >> qubit) & 1
        z += float(p) * (1.0 if bit == 0 else -1.0)
    return z


def _loss_from_outputs(
    outputs: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    loss_name: LossName,
    eps: float = 1e-10,
) -> float:
    if loss_name == "softmax_nll":
        return batch_loss(outputs["class_probs"], y_true, eps=eps)

    y_true = y_true.astype(np.int64)
    z_expect = outputs["z_expectations"]
    if loss_name == "cross_entropy_expectation":
        p1 = _expectation_to_p1(z_expect, eps=eps)
        losses = -(y_true * np.log(p1) + (1 - y_true) * np.log(1.0 - p1))
        return float(np.mean(losses))

    # Binary hinge loss on margin score s(x) = -<Z>; y in {-1, +1}.
    y_pm = 2 * y_true - 1
    score = -z_expect
    margins = y_pm * score
    return float(np.mean(np.maximum(0.0, 1.0 - margins)))


def _predict_from_outputs(
    outputs: dict[str, np.ndarray],
    *,
    decision_rule: DecisionRule,
) -> np.ndarray:
    if decision_rule == "argmax":
        return predict_batch(outputs["class_probs"])
    return (outputs["z_expectations"] < 0.0).astype(np.int64)
