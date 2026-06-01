r"""Quantum-kernel matrix construction and validation.

Builds feature-map statevectors for Nim states and computes the kernel
``k(x, x') = |<psi(x)|psi(x')>|^2`` either exactly (statevector inner product)
or via finite-shot binomial sampling.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from qiskit.quantum_info import Statevector
from sklearn.utils import check_array
from sklearn.utils.validation import check_symmetric

from qml_project.nim.data import normalise_states
from qml_project.nim.encoding import EncodingName, SymmetryMode, build_encoding_circuit
from qml_project.nim.game import nim_sum
from qml_project.nim.state_utils import state_tuple_from_array

KernelEstimatorMode = Literal["exact_statevector", "shot_binomial"]
KernelBackend = Literal["manual", "qiskit_fidelity"]

_state_key = state_tuple_from_array


def _build_statevector(
    state: np.ndarray | Sequence[int],
    *,
    encoding: EncodingName,
    M: int = 7,
    bits_per_heap: int = 3,
    iqp_reps: int = 2,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Build a feature-map statevector for one Nim state."""
    circuit = build_encoding_circuit(
        encoding,
        _state_key(state),
        M=M,
        bits_per_heap=bits_per_heap,
        iqp_reps=iqp_reps,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,
    )
    return Statevector.from_instruction(circuit).data


def _prepare_angle_features(
    states: np.ndarray,
    *,
    M: int,
    symmetry: SymmetryMode,
    include_nim_sum: bool = True,
) -> np.ndarray:
    """Map integer Nim heaps to angle features used by angle encoding."""
    arr = np.asarray(states, dtype=np.int32)
    if symmetry == "canonical":
        arr = np.sort(arr, axis=1)
    elif symmetry != "none":
        raise ValueError(
            f"kernel_backend='qiskit_fidelity' currently supports symmetry in ('none', 'canonical'), got {symmetry!r}"
        )
    thetas = normalise_states(arr, M_max=M) * np.pi
    if not include_nim_sum:
        return thetas
    ns = np.array(
        [float(nim_sum(tuple(int(x) for x in row))) for row in arr],
        dtype=np.float64,
    )
    ns_col = (ns * np.pi / float(M)).reshape(-1, 1)
    return np.hstack([thetas, ns_col])


def _quantum_kernel_matrix_qiskit_fidelity(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    M: int,
    symmetry: SymmetryMode,
    include_nim_sum: bool,
) -> np.ndarray:
    """Experimental qiskit-machine-learning backend for angle kernels."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit_machine_learning.kernels import (  # pyright: ignore[reportMissingImports]
            FidelityStatevectorKernel,
        )
    except ImportError as exc:
        raise ImportError(
            "kernel_backend='qiskit_fidelity' requires qiskit-machine-learning."
        ) from exc

    n_features = 4 if include_nim_sum else 3
    params = ParameterVector("x", n_features)
    feature_map = QuantumCircuit(n_features)
    for i in range(n_features):
        feature_map.ry(params[i], i)
    kernel = FidelityStatevectorKernel(feature_map=feature_map)
    x_feat = _prepare_angle_features(
        X, M=M, symmetry=symmetry, include_nim_sum=include_nim_sum
    )
    y_feat = _prepare_angle_features(
        Y, M=M, symmetry=symmetry, include_nim_sum=include_nim_sum
    )
    return np.asarray(kernel.evaluate(x_vec=x_feat, y_vec=y_feat), dtype=np.float64)


def quantum_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    encoding: EncodingName,
    M: int = 7,
    bits_per_heap: int = 3,
    iqp_reps: int = 2,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
    estimator_mode: KernelEstimatorMode = "exact_statevector",
    kernel_backend: KernelBackend = "manual",
    shots: int = 1024,
    seed: int = 42,
    validate: bool = True,
) -> np.ndarray:
    """Compute the quantum kernel matrix between two state sets.

    Notes
    -----
    Uses exact statevector overlap:
    ``k(x, x') = |<psi(x)|psi(x')>|^2``.
    """
    X = np.asarray(X, dtype=np.int32)
    Y = np.asarray(Y, dtype=np.int32)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays of raw heap states.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of heaps/features.")

    if kernel_backend == "qiskit_fidelity":
        if encoding != "angle":
            raise ValueError("kernel_backend='qiskit_fidelity' currently supports encoding='angle' only")
        if estimator_mode != "exact_statevector":
            raise ValueError("kernel_backend='qiskit_fidelity' requires estimator_mode='exact_statevector'")
        K = _quantum_kernel_matrix_qiskit_fidelity(
            X, Y, M=M, symmetry=symmetry, include_nim_sum=bool(include_nim_sum)
        )
    elif kernel_backend == "manual":
        cache: dict[tuple[int, ...], np.ndarray] = {}

        def get_sv(state: np.ndarray) -> np.ndarray:
            key = _state_key(state)
            if key not in cache:
                cache[key] = _build_statevector(
                    key,
                    encoding=encoding,
                    M=M,
                    bits_per_heap=bits_per_heap,
                    iqp_reps=iqp_reps,
                    include_nim_sum=include_nim_sum,
                    symmetry=symmetry,
                )
            return cache[key]

        sv_x = [get_sv(X[i]) for i in range(X.shape[0])]
        sv_y = [get_sv(Y[j]) for j in range(Y.shape[0])]

        if estimator_mode == "shot_binomial" and shots < 1:
            raise ValueError("shots must be >= 1 for shot_binomial estimator")

        K = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        for i, a in enumerate(sv_x):
            overlaps = np.array([np.vdot(a, b) for b in sv_y], dtype=np.complex128)
            probs = np.abs(overlaps) ** 2
            if estimator_mode == "exact_statevector":
                K[i] = probs
            elif estimator_mode == "shot_binomial":
                K[i] = rng.binomial(int(shots), np.clip(probs, 0.0, 1.0)) / float(shots)
            else:
                raise ValueError(f"Unknown estimator_mode: {estimator_mode}")
    else:
        raise ValueError(f"Unknown kernel_backend: {kernel_backend}")

    is_square = X.shape == Y.shape and np.array_equal(X, Y)
    if is_square and estimator_mode == "shot_binomial":
        K = 0.5 * (K + K.T)

    if validate:
        _validate_kernel_matrix(
            K,
            square=is_square,
            estimator_mode=estimator_mode,
        )
    return K


def _validate_kernel_matrix(
    K: np.ndarray,
    *,
    square: bool,
    estimator_mode: KernelEstimatorMode,
) -> None:
    """Sanity-check kernel matrix shape, finiteness, symmetry, and diagonal before SVC fit."""
    try:
        K_checked = check_array(
            K,
            ensure_2d=True,
            allow_nd=False,
        )
    except ValueError as exc:
        raise ValueError("Kernel matrix must be a finite 2D float array.") from exc
    K_checked = np.asarray(K_checked, dtype=np.float64)
    if not np.all(np.isfinite(K_checked)):
        raise ValueError("Kernel matrix must be a finite 2D float array.")

    if square:
        try:
            check_symmetric(K_checked, tol=1e-8, raise_warning=False, raise_exception=True)
        except ValueError as exc:
            raise ValueError("Train kernel matrix must be symmetric.") from exc
        diag = np.diag(K_checked)
        if estimator_mode == "exact_statevector":
            if not np.allclose(diag, 1.0, atol=1e-8):
                raise ValueError("Exact overlap train kernel must have unit diagonal.")
        else:
            if np.any(diag < 0.7):
                raise ValueError("Shot-estimated train kernel diagonal too far from 1.")


__all__ = [
    "KernelBackend",
    "KernelEstimatorMode",
    "quantum_kernel_matrix",
]
