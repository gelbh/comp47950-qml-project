"""QSVM device payload, refit, overlap circuits, and counts decoding.

Section 10 submits the QSVM winner to IBM Runtime. We refit on a small
training budget (``DEVICE_TRAIN_SIZE = 50`` by default) so the post-fit
support-vector count stays tractable on free-tier shots and the low-data
hypothesis is preserved.

Circuit construction notes:

- **Amplitude** uses ``StatePreparation`` (not ``Initialize``) so
  ``.inverse()`` is well-defined — ``Initialize`` includes a reset and
  cannot be inverted cleanly.
- **Angle** applies per-qubit ``RY(θ_b)`` then ``RY(-θ_a)`` on the vacuum,
  matching ``|⟨ψ(a)|ψ(b)⟩|²`` for the product ``RY`` feature map.

**Simulation vs device.** Training and overlap circuits use the same
``encoding``, ``include_nim_sum``, and ``symmetry`` as the MLflow winner
row so hardware inference tests the *same* decision rule under finite
shots, not the noiseless exact statevector kernel used during sim sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.primitives.containers.sampler_pub import SamplerPub

from qml_project.nim.encoding import (
    EncodingName,
    SymmetryMode,
    amplitude_vector,
    angle_parameters,
)
from qml_project.nim.state_utils import state_tuple_from_array
from qml_project.qsvm import QuantumKernelSVMModel, fit_quantum_kernel_svm


@dataclass
class QSVMDevicePayload:
    """Serialisable QSVM artefact ready for device submission.

    Stores the fitted ``QuantumKernelSVMModel`` along with the post-fit
    support-vector indices and dual coefficients we need to reconstruct
    the decision function on the device kernel row. The canonical SVC
    ``predict`` path would require a kernel row against the **full**
    training set; using ``support_`` / ``dual_coef_`` directly lets us
    compute overlaps only for the SVs we actually need, cutting device
    circuits by ``|train| / |SV|``.
    """

    variant_id: str
    encoding: EncodingName
    model: QuantumKernelSVMModel
    X_train_raw: np.ndarray
    y_train: np.ndarray
    sv_indices: np.ndarray
    support_vectors_raw: np.ndarray
    dual_coef: np.ndarray
    intercept: float
    class_labels: np.ndarray
    c_svc: float
    symmetry: SymmetryMode
    bits_per_heap: int
    iqp_reps: int
    include_nim_sum: bool
    M: int
    train_size_used: int
    refit_seed: int = 0
    refit_balanced_accuracy: float | None = None


def refit_qsvm_for_device(
    *,
    winner_row: Mapping[str, Any],
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    train_size: int = 50,
    seed: int = 0,
) -> QSVMDevicePayload:
    """Refit a QSVM at the winner's hyperparameters and extract SV metadata."""
    from sklearn.metrics import balanced_accuracy_score

    from qml_project.nim.data import training_subsets

    encoding = str(winner_row["encoding"])
    symmetry = str(winner_row.get("symmetry", "none"))
    c_svc = float(winner_row.get("c_svc", 1.0))
    bits_per_heap = int(winner_row.get("bits_per_heap", 3))
    iqp_reps = int(winner_row.get("iqp_reps", 2))
    include_nim_sum = bool(winner_row.get("include_nim_sum", True))

    # ``training_subsets`` skips any size ``>= len(X_train)`` and stores the
    # full set under ``"full"``. Clamp when the requested refit budget meets
    # or exceeds the training set (e.g. ``train_size=215`` on the 215-state
    # training split).
    _X_train_arr = np.asarray(X_train_raw, dtype=np.int32)
    _n_train_full = int(len(_X_train_arr))
    _effective_size = int(train_size)
    if _effective_size >= _n_train_full:
        _effective_size = _n_train_full
        subsets = training_subsets(
            _X_train_arr,
            np.asarray(y_train),
            sizes=[],
            random_state=int(seed),
        )
        subset = subsets["full"]
    else:
        subsets = training_subsets(
            _X_train_arr,
            np.asarray(y_train),
            sizes=[_effective_size],
            random_state=int(seed),
        )
        subset = subsets[_effective_size]

    qsvm, _train_time, _kernel_time = fit_quantum_kernel_svm(
        subset.X,
        subset.y,
        encoding=encoding,  # type: ignore[arg-type]
        symmetry=symmetry,  # type: ignore[arg-type]
        M=7,
        bits_per_heap=bits_per_heap,
        iqp_reps=iqp_reps,
        include_nim_sum=include_nim_sum,
        random_state=int(seed),
        c_svc=c_svc,
    )

    sv_indices = np.asarray(qsvm.model.support_, dtype=np.int64)
    support_vectors_raw = qsvm.X_train_raw[sv_indices].copy()
    dual_coef = np.asarray(qsvm.model.dual_coef_, dtype=np.float64).ravel()
    intercept = float(qsvm.model.intercept_[0])
    class_labels = np.asarray(qsvm.model.classes_)

    refit_accuracy: float | None = None
    if X_test_raw is not None and y_test is not None:
        preds = qsvm.predict_states(np.asarray(X_test_raw, dtype=np.int32))
        refit_accuracy = float(
            balanced_accuracy_score(np.asarray(y_test), preds)
        )

    return QSVMDevicePayload(
        variant_id=str(winner_row.get("variant_id", "")),
        encoding=encoding,  # type: ignore[arg-type]
        model=qsvm,
        X_train_raw=np.asarray(subset.X, dtype=np.int32),
        y_train=np.asarray(subset.y),
        sv_indices=sv_indices,
        support_vectors_raw=np.asarray(support_vectors_raw, dtype=np.int32),
        dual_coef=dual_coef,
        intercept=intercept,
        class_labels=class_labels,
        c_svc=c_svc,
        symmetry=symmetry,  # type: ignore[arg-type]
        bits_per_heap=bits_per_heap,
        iqp_reps=iqp_reps,
        include_nim_sum=include_nim_sum,
        M=7,
        train_size_used=int(_effective_size),
        refit_seed=int(seed),
        refit_balanced_accuracy=refit_accuracy,
    )


def _amplitude_overlap_circuit(
    state_a: Sequence[int],
    state_b: Sequence[int],
    *,
    M: int,
    include_nim_sum: bool,
    symmetry: SymmetryMode,
) -> QuantumCircuit:
    """Build an overlap circuit U(a)† U(b)|0⟩ for amplitude encoding.

    Measured all-zeros probability equals ``|⟨ψ(a)|ψ(b)⟩|²``. We use
    ``StatePreparation`` (gate-only) rather than ``initialize`` so
    ``.inverse()`` is well-defined — ``Initialize`` bundles a reset and
    cannot be cleanly inverted on a hardware target.
    """
    sym: SymmetryMode = symmetry if symmetry in ("none", "canonical") else "none"
    vec_a = amplitude_vector(
        state_tuple_from_array(state_a),
        M=M,
        include_nim_sum=include_nim_sum,
        symmetry=sym,
    )
    vec_b = amplitude_vector(
        state_tuple_from_array(state_b),
        M=M,
        include_nim_sum=include_nim_sum,
        symmetry=sym,
    )
    n_qubits = int(np.log2(vec_a.size))
    if 2**n_qubits != vec_a.size:
        raise ValueError("amplitude vector length must be a power of two")
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.append(StatePreparation(vec_b), range(n_qubits))
    qc.append(StatePreparation(vec_a).inverse(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def _angle_overlap_circuit(
    state_a: Sequence[int],
    state_b: Sequence[int],
    *,
    M: int,
    include_nim_sum: bool,
    symmetry: SymmetryMode,
) -> QuantumCircuit:
    """Build overlap circuit for angle encoding (product of single-qubit RY).

    Feature map :math:`U(\\mathbf x)|0\\rangle = \\bigotimes_i R_Y(\\theta_i(x))|0\\rangle`
    with ``theta`` from :func:`~qml_project.nim.encoding.angle_parameters`.
    Then :math:`|\\langle 0^{\\otimes n} | U(\\mathbf a)^\\dagger U(\\mathbf b) |0\\rangle|^2
    = |\\langle\\psi(\\mathbf a)|\\psi(\\mathbf b)\\rangle|^2`, estimated by the
    all-zeros measurement probability (same decoding as amplitude overlaps).
    """
    sym: SymmetryMode = symmetry if symmetry in ("none", "canonical") else "none"
    theta_a = angle_parameters(
        state_tuple_from_array(state_a),
        M=M,
        include_nim_sum=include_nim_sum,
        symmetry=sym,
    )
    theta_b = angle_parameters(
        state_tuple_from_array(state_b),
        M=M,
        include_nim_sum=include_nim_sum,
        symmetry=sym,
    )
    n_qubits = int(theta_a.size)
    if int(theta_b.size) != n_qubits:
        raise ValueError("angle overlap: mismatched parameter counts for the two states")
    qc = QuantumCircuit(n_qubits, n_qubits)
    for q in range(n_qubits):
        qc.ry(float(theta_b[q]), q)
        qc.ry(float(-theta_a[q]), q)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def build_qsvm_device_pubs(
    payload: QSVMDevicePayload,
    X_test_raw: np.ndarray,
    *,
    shots: int = 1024,
) -> tuple[list[SamplerPub], dict[str, Any]]:
    """Build one overlap PUB per (test state, support vector) pair.

    The metadata dict records ``n_test``, ``n_sv``, and a ``pair_index``
    array of shape ``(n_test * n_sv, 2)`` so ``decode_qsvm_counts`` can
    reassemble the device kernel matrix in the correct order.
    """
    enc = str(payload.encoding)
    if enc not in ("amplitude", "angle"):
        raise NotImplementedError(
            f"device QSVM supports encoding='amplitude' or 'angle' only; "
            f"winner uses {payload.encoding!r}."
        )
    X_test = np.asarray(X_test_raw, dtype=np.int32)
    n_test = int(X_test.shape[0])
    n_sv = int(payload.support_vectors_raw.shape[0])
    pubs: list[SamplerPub] = []
    pair_index = np.empty((n_test * n_sv, 2), dtype=np.int64)
    for t in range(n_test):
        for s in range(n_sv):
            if enc == "amplitude":
                qc = _amplitude_overlap_circuit(
                    X_test[t],
                    payload.support_vectors_raw[s],
                    M=payload.M,
                    include_nim_sum=payload.include_nim_sum,
                    symmetry=payload.symmetry,
                )
            else:
                qc = _angle_overlap_circuit(
                    X_test[t],
                    payload.support_vectors_raw[s],
                    M=payload.M,
                    include_nim_sum=payload.include_nim_sum,
                    symmetry=payload.symmetry,
                )
            pubs.append(SamplerPub(circuit=qc, shots=int(shots)))
            pair_index[t * n_sv + s] = (t, s)
    metadata = {
        "n_test": n_test,
        "n_sv": n_sv,
        "pair_index": pair_index,
        "shots": int(shots),
    }
    return pubs, metadata


def decode_qsvm_counts(
    counts_list: Sequence[Mapping[str, int]],
    payload: QSVMDevicePayload,
    metadata: Mapping[str, Any],
) -> np.ndarray:
    """Reconstruct predictions from per-(test, SV) overlap counts.

    Estimates the kernel row ``k(x_t, x_sv)`` as the all-zeros probability
    of each overlap circuit, then computes the SVC decision function
    ``f(x_t) = Σ α_i y_i k(x_t, x_sv_i) + b`` using the SVs only.
    """
    n_test = int(metadata["n_test"])
    n_sv = int(metadata["n_sv"])
    if len(counts_list) != n_test * n_sv:
        raise ValueError(
            f"counts_list has {len(counts_list)} entries; expected "
            f"n_test*n_sv = {n_test * n_sv}"
        )
    # Determine qubit count from the first non-empty counts dict so we
    # know how many zeros the "all-zeros" outcome is.
    n_qubits: int | None = None
    for c in counts_list:
        if c:
            n_qubits = len(next(iter(c)))
            break
    if n_qubits is None:
        raise ValueError("all counts dictionaries are empty; cannot decode")
    zero_bitstring = "0" * n_qubits

    K_row = np.zeros((n_test, n_sv), dtype=np.float64)
    pair_index = np.asarray(metadata["pair_index"], dtype=np.int64)
    for k, counts in enumerate(counts_list):
        total = float(sum(counts.values()))
        if total <= 0:
            p_zero = 0.0
        else:
            p_zero = float(counts.get(zero_bitstring, 0)) / total
        t, s = int(pair_index[k, 0]), int(pair_index[k, 1])
        K_row[t, s] = p_zero

    decision = K_row @ payload.dual_coef + payload.intercept
    preds = np.where(
        decision >= 0.0, payload.class_labels[1], payload.class_labels[0]
    )
    return np.asarray(preds)


__all__ = [
    "QSVMDevicePayload",
    "build_qsvm_device_pubs",
    "decode_qsvm_counts",
    "refit_qsvm_for_device",
]
