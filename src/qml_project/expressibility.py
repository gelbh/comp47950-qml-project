"""
Expressibility and trainability diagnostics for variational circuits.

Implements three diagnostics used in the QML design stage:
  - Expressibility via KL divergence to the Haar-random fidelity distribution
    (Sim et al., 2019).
  - Entangling capability via the Meyer-Wallach global entanglement measure.
  - Barren plateau screening via gradient-variance vs depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace, state_fidelity

from qml_project.circuit import VariationalClassifier, build_circuit


@dataclass(frozen=True)
class ExpressibilityMetrics:
    """Expressibility metrics for one circuit configuration."""

    kl_divergence_to_haar: float
    mean_fidelity: float
    std_fidelity: float
    n_pairs: int
    n_bins: int


@dataclass(frozen=True)
class EntanglingMetrics:
    """Entangling capability metrics for one circuit configuration."""

    meyer_wallach_mean: float
    meyer_wallach_std: float
    n_samples: int


def _strip_measurements(vc: VariationalClassifier):
    """Return a copy of the circuit with final measurements removed."""
    return vc.circuit.remove_final_measurements(inplace=False)


def _random_parameter_bindings(
    vc: VariationalClassifier,
    rng: np.random.Generator,
    *,
    feature_range: tuple[float, float],
    trainable_range: tuple[float, float],
) -> dict:
    """Sample one random assignment for feature and trainable parameters."""
    x_low, x_high = feature_range
    w_low, w_high = trainable_range
    x_vals = rng.uniform(x_low, x_high, len(vc.feature_params))
    w_vals = rng.uniform(w_low, w_high, len(vc.trainable_params))

    bindings: dict = {}
    for idx, param in enumerate(vc.feature_params):
        bindings[param] = float(x_vals[idx])
    for idx, param in enumerate(vc.trainable_params):
        bindings[param] = float(w_vals[idx])
    return bindings


def _sample_statevectors(
    vc: VariationalClassifier,
    *,
    n_samples: int,
    seed: int,
    feature_range: tuple[float, float] = (0.0, np.pi),
    trainable_range: tuple[float, float] = (-np.pi, np.pi),
) -> list[Statevector]:
    """Sample random output states from one variational circuit family."""
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 for fidelity-based diagnostics.")

    rng = np.random.default_rng(seed)
    qc = _strip_measurements(vc)
    states: list[Statevector] = []

    for _ in range(n_samples):
        bindings = _random_parameter_bindings(
            vc,
            rng,
            feature_range=feature_range,
            trainable_range=trainable_range,
        )
        bound = qc.assign_parameters(bindings, inplace=False)
        states.append(Statevector.from_instruction(bound))
    return states


def _haar_bin_probabilities(n_qubits: int, bin_edges: np.ndarray) -> np.ndarray:
    """
    Return Haar fidelity-bin probabilities for pure states in dimension d=2^n.

    If F is fidelity between two Haar-random pure states, then:
      p(F) = (d - 1) * (1 - F)^(d - 2), F in [0, 1]
      CDF(F) = 1 - (1 - F)^(d - 1)
    """
    d = 2**n_qubits
    left = bin_edges[:-1]
    right = bin_edges[1:]
    probs = (1.0 - left) ** (d - 1) - (1.0 - right) ** (d - 1)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("Invalid Haar bin probabilities.")
    return probs / total


def estimate_expressibility(
    vc: VariationalClassifier,
    *,
    n_samples: int = 256,
    n_pairs: int = 2048,
    n_bins: int = 75,
    seed: int = 42,
) -> ExpressibilityMetrics:
    """
    Estimate expressibility as KL(empirical fidelity || Haar fidelity).

    Lower KL indicates the random state distribution induced by the ansatz is
    closer to Haar-random and therefore more expressive (Sim et al., 2019).
    """
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1.")
    if n_bins < 5:
        raise ValueError("n_bins must be >= 5.")

    states = _sample_statevectors(vc, n_samples=n_samples, seed=seed)
    rng = np.random.default_rng(seed + 1)

    fidelities = np.empty(n_pairs, dtype=np.float64)
    n_states = len(states)
    for i in range(n_pairs):
        a = int(rng.integers(0, n_states))
        b = int(rng.integers(0, n_states - 1))
        if b >= a:
            b += 1
        fidelities[i] = float(state_fidelity(states[a], states[b]))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    empirical_hist, _ = np.histogram(fidelities, bins=bin_edges, density=False)
    empirical = empirical_hist.astype(np.float64)
    empirical /= empirical.sum()

    haar = _haar_bin_probabilities(vc.n_qubits, bin_edges)
    eps = 1e-12
    p = np.clip(empirical, eps, 1.0)
    q = np.clip(haar, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))

    return ExpressibilityMetrics(
        kl_divergence_to_haar=kl,
        mean_fidelity=float(np.mean(fidelities)),
        std_fidelity=float(np.std(fidelities)),
        n_pairs=n_pairs,
        n_bins=n_bins,
    )


def _meyer_wallach_q(state: Statevector, n_qubits: int) -> float:
    """Compute Meyer-Wallach global entanglement Q in [0, 1] for pure states."""
    if n_qubits < 2:
        return 0.0

    purities: list[float] = []
    full = list(range(n_qubits))
    for qubit in range(n_qubits):
        trace_out = [q for q in full if q != qubit]
        reduced = partial_trace(state, trace_out)
        rho = np.asarray(reduced.data, dtype=np.complex128)
        purity = float(np.real(np.trace(rho @ rho)))
        purities.append(purity)

    return float(2.0 * (1.0 - np.mean(purities)))


def estimate_entangling_capability(
    vc: VariationalClassifier,
    *,
    n_samples: int = 256,
    seed: int = 42,
) -> EntanglingMetrics:
    """Estimate entangling capability by sampling Meyer-Wallach Q values."""
    states = _sample_statevectors(vc, n_samples=n_samples, seed=seed)
    q_values = np.array(
        [_meyer_wallach_q(state, vc.n_qubits) for state in states],
        dtype=np.float64,
    )
    return EntanglingMetrics(
        meyer_wallach_mean=float(np.mean(q_values)),
        meyer_wallach_std=float(np.std(q_values)),
        n_samples=n_samples,
    )


def _z0_operator(n_qubits: int) -> SparsePauliOp:
    """Pauli-Z operator on qubit 0 (Qiskit little-endian convention)."""
    label = "I" * (n_qubits - 1) + "Z"
    return SparsePauliOp.from_list([(label, 1.0)])


def _mean_z0_cost(
    vc: VariationalClassifier,
    qc_no_measure,
    x_batch: np.ndarray,
    theta: np.ndarray,
    z0: SparsePauliOp,
) -> float:
    """
    Mean <Z0> over a small batch of input angles.

    This cost is intentionally simple so gradient scaling with depth can be
    measured without confounding training dynamics.
    """
    vals: list[float] = []
    for x in x_batch:
        bindings = {
            **{p: float(v) for p, v in zip(vc.feature_params, x)},
            **{p: float(v) for p, v in zip(vc.trainable_params, theta)},
        }
        bound = qc_no_measure.assign_parameters(bindings, inplace=False)
        state = Statevector.from_instruction(bound)
        exp_val = state.expectation_value(z0)
        vals.append(float(np.real(exp_val)))
    return float(np.mean(vals))


def gradient_variance_vs_depth(
    *,
    ansatz: str,
    n_qubits: int,
    n_features: int,
    n_classes: int,
    depths: list[int],
    cz_strategy: str = "linear",
    cz_seed: int = 42,
    n_initializations: int = 20,
    batch_size: int = 8,
    finite_diff_eps: float = 1e-3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Estimate gradient variance as a function of circuit depth.

    For each depth, random initialisations are sampled and central finite-
    difference gradients of mean <Z0> are computed. A rapidly decaying gradient
    variance with depth is a practical barren-plateau warning signal [13].
    """
    if n_initializations < 2:
        raise ValueError("n_initializations must be >= 2.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if finite_diff_eps <= 0.0:
        raise ValueError("finite_diff_eps must be positive.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for depth in depths:
        vc = build_circuit(
            n_qubits=n_qubits,
            n_features=n_features,
            n_classes=n_classes,
            n_layers=depth,
            cz_strategy=cz_strategy,
            cz_seed=cz_seed,
            ansatz=ansatz,  # type: ignore[arg-type]
        )
        qc_no_measure = _strip_measurements(vc)
        z0 = _z0_operator(vc.n_qubits)

        grad_vectors = np.zeros((n_initializations, vc.n_trainable), dtype=np.float64)
        for init_idx in range(n_initializations):
            theta = rng.uniform(-np.pi, np.pi, vc.n_trainable)
            x_batch = rng.uniform(0.0, np.pi, size=(batch_size, vc.n_features))

            for param_idx in range(vc.n_trainable):
                theta_plus = theta.copy()
                theta_minus = theta.copy()
                theta_plus[param_idx] += finite_diff_eps
                theta_minus[param_idx] -= finite_diff_eps

                c_plus = _mean_z0_cost(vc, qc_no_measure, x_batch, theta_plus, z0)
                c_minus = _mean_z0_cost(vc, qc_no_measure, x_batch, theta_minus, z0)
                grad_vectors[init_idx, param_idx] = (c_plus - c_minus) / (
                    2.0 * finite_diff_eps
                )

        rows.append(
            {
                "ansatz": ansatz,
                "depth": depth,
                "n_trainable": vc.n_trainable,
                "gradient_variance_mean": float(np.mean(np.var(grad_vectors, axis=1))),
                "gradient_variance_std": float(np.std(np.var(grad_vectors, axis=1))),
                "gradient_abs_mean": float(np.mean(np.abs(grad_vectors))),
                "n_initializations": n_initializations,
                "batch_size": batch_size,
            }
        )

    return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)


def compare_ansatz_expressibility(
    *,
    ansatze: list[str],
    n_qubits: int,
    n_features: int,
    n_classes: int,
    n_layers: int,
    cz_strategy: str = "linear",
    cz_seed: int = 42,
    n_samples: int = 256,
    n_pairs: int = 2048,
    n_bins: int = 75,
    seed: int = 42,
) -> pd.DataFrame:
    """Tabulate expressibility and entangling capability for each ansatz."""
    rows: list[dict[str, float | int | str]] = []

    for i, ansatz in enumerate(ansatze):
        vc = build_circuit(
            n_qubits=n_qubits,
            n_features=n_features,
            n_classes=n_classes,
            n_layers=n_layers,
            cz_strategy=cz_strategy,
            cz_seed=cz_seed,
            ansatz=ansatz,  # type: ignore[arg-type]
        )
        expr = estimate_expressibility(
            vc,
            n_samples=n_samples,
            n_pairs=n_pairs,
            n_bins=n_bins,
            seed=seed + i,
        )
        ent = estimate_entangling_capability(
            vc,
            n_samples=n_samples,
            seed=seed + 100 + i,
        )
        rows.append(
            {
                "ansatz": ansatz,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "n_trainable": vc.n_trainable,
                "kl_divergence_to_haar": expr.kl_divergence_to_haar,
                "mean_fidelity": expr.mean_fidelity,
                "std_fidelity": expr.std_fidelity,
                "meyer_wallach_mean": ent.meyer_wallach_mean,
                "meyer_wallach_std": ent.meyer_wallach_std,
            }
        )

    return pd.DataFrame(rows).sort_values("kl_divergence_to_haar").reset_index(drop=True)
