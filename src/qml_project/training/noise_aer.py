"""Qiskit Aer noise models, readout mitigation, and ZNE helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from qml_project.training.types import VqcAnsatzHypothesis


def create_depolarizing_noise_model(
    cz_error_rate: float = 0.01,
    single_gate_error_rate: float = 0.0,
    readout_error_rate: float = 0.0,
) -> Any:
    """
    Create a depolarizing noise model for simulation.

    Parameters
    ----------
    cz_error_rate : float
        Depolarizing error probability on CZ (two-qubit) gates.
    single_gate_error_rate : float
        Depolarizing error probability on single-qubit gates (rx, rz).

    Returns
    -------
    qiskit_aer.noise.NoiseModel
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise_model = NoiseModel()

    if cz_error_rate > 0:
        error_cz = depolarizing_error(cz_error_rate, 2)
        noise_model.add_all_qubit_quantum_error(error_cz, ["cz"])

    if single_gate_error_rate > 0:
        error_1q = depolarizing_error(single_gate_error_rate, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ["rx", "rz"])

    if readout_error_rate > 0:
        from qiskit_aer.noise import ReadoutError

        p = float(readout_error_rate)
        ro = ReadoutError([[1.0 - p, p], [p, 1.0 - p]])
        noise_model.add_all_qubit_readout_error(ro)

    return noise_model


def create_noisy_sampler(
    noise_model: Any,
    seed: int = 42,
) -> Any:
    """
    Create a V2 sampler backed by Qiskit Aer with a noise model.

    Parameters
    ----------
    noise_model
        A ``qiskit_aer.noise.NoiseModel``.
    seed : int
        Simulator random seed for reproducibility.

    Returns
    -------
    A sampler compatible with ``StatevectorSampler``'s V2 interface.
    """
    from qiskit_aer.primitives import SamplerV2

    return SamplerV2(
        options={
            "backend_options": {
                "noise_model": noise_model,
                "seed_simulator": seed,
            },
        },
        seed=seed,
    )


def default_vqc_ansatz_hypotheses() -> dict[str, VqcAnsatzHypothesis]:
    """
    Return explicit, report-ready hypotheses for built-in ansatze.

    These hypotheses are used by notebook narrative and MLflow metadata.
    """
    return {
        "basic_block": VqcAnsatzHypothesis(
            ansatz="basic_block",
            hypothesis=(
                "RX(pi/2)-RZ-RX(pi/2) layers bias each qubit toward phase rotations "
                "that can represent parity-sensitive boundaries with modest depth."
            ),
            expected_strength=(
                "Higher expressivity at the same depth; stronger ceiling on clean simulation."
            ),
            primary_risk=(
                "May become noise-sensitive because each block has extra 1-qubit gates."
            ),
        ),
        "ry_rz": VqcAnsatzHypothesis(
            ansatz="ry_rz",
            hypothesis=(
                "Lean RY-RZ blocks reduce gate count so training under noise remains "
                "more stable, even if clean-sim expressivity is slightly lower."
            ),
            expected_strength=(
                "Holds up better under depolarising/readout noise at fixed shots."
            ),
            primary_risk=(
                "Potential underfitting if shallow depth cannot represent full Nim-sum geometry."
            ),
        ),
    }


def build_assignment_matrix_from_symmetric_readout_error(
    *,
    n_qubits: int,
    readout_error_rate: float,
) -> np.ndarray:
    """Construct full assignment matrix A where p_obs = A @ p_true."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    p = float(readout_error_rate)
    if p < 0.0 or p >= 0.5:
        raise ValueError("readout_error_rate must be in [0, 0.5).")
    one_q = np.array([[1.0 - p, p], [p, 1.0 - p]], dtype=np.float64)
    mat = one_q
    for _ in range(n_qubits - 1):
        mat = np.kron(one_q, mat)
    return mat


def mitigate_readout_prob_vector(
    observed_probs: np.ndarray,
    assignment_matrix: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply linear-inversion readout correction with clipping and renormalisation.
    """
    obs = np.asarray(observed_probs, dtype=np.float64)
    A = np.asarray(assignment_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("assignment_matrix must be square.")
    if obs.ndim != 1 or obs.shape[0] != A.shape[0]:
        raise ValueError("observed_probs shape must match assignment_matrix.")
    corrected = np.linalg.pinv(A) @ obs
    corrected = np.clip(corrected, 0.0, None)
    total = float(np.sum(corrected))
    if total <= eps:
        return np.ones_like(corrected) / float(corrected.size)
    return corrected / total


def zne_extrapolate_to_zero(
    scales: Sequence[float],
    values: Sequence[float],
    *,
    degree: int = 1,
) -> float:
    """Extrapolate scalar metric from scaled-noise values to zero noise."""
    x = np.asarray(scales, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.size != y.size or x.size == 0:
        raise ValueError("scales and values must have the same non-zero length.")
    deg = int(min(max(1, degree), x.size - 1))
    coeff = np.polyfit(x, y, deg=deg)
    return float(np.polyval(coeff, 0.0))


def _zne_extrapolate_outputs(
    outputs_per_scale: Sequence[dict[str, np.ndarray]],
    *,
    scales: Sequence[float],
    degree: int,
) -> dict[str, np.ndarray]:
    """Extrapolate class probabilities and Z expectations to zero noise."""
    if len(outputs_per_scale) == 0:
        raise ValueError("outputs_per_scale cannot be empty.")
    x = np.asarray(scales, dtype=np.float64)
    cp_stack = np.stack([o["class_probs"] for o in outputs_per_scale], axis=0)
    z_stack = np.stack([o["z_expectations"] for o in outputs_per_scale], axis=0)
    deg = int(min(max(1, degree), len(outputs_per_scale) - 1))

    n_samples, n_classes = cp_stack.shape[1], cp_stack.shape[2]
    cp0 = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_classes):
            coeff = np.polyfit(x, cp_stack[:, i, j], deg=deg)
            cp0[i, j] = float(np.polyval(coeff, 0.0))
    cp0 = np.clip(cp0, 0.0, None)
    cp0_sum = cp0.sum(axis=1, keepdims=True)
    cp0_sum = np.where(cp0_sum <= 1e-12, 1.0, cp0_sum)
    cp0 = cp0 / cp0_sum

    z0 = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        coeff = np.polyfit(x, z_stack[:, i], deg=deg)
        z0[i] = float(np.polyval(coeff, 0.0))
    z0 = np.clip(z0, -1.0, 1.0)

    return {"class_probs": cp0, "z_expectations": z0}
