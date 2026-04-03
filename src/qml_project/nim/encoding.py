"""Nim-specific QML input encodings and pilot go/no-go selection.

This module implements the staged encoding strategy from the project plan:

1. **Angle encoding** on 3 qubits (default starter): ``theta_i = h_i * pi / M``.
2. **Amplitude encoding** pilot: compact qubit count with state-preparation cost.
3. **Binary encoding** on 9 qubits: three bits per heap with bit-position
   entanglement aligned to Nim-sum XOR.
4. **Parity / IQP-style feature map** to emphasise XOR-like structure.

Encoding rationale follows the report references [18] and [20].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

from qml_project.nim.game import NimState, nim_sum

EncodingName = Literal["angle", "amplitude", "binary", "iqp_parity"]
SymmetryMode = Literal["none", "canonical", "equivariant"]
ENCODING_CANDIDATES: tuple[EncodingName, ...] = (
    "angle",
    "amplitude",
    "binary",
    "iqp_parity",
)


@dataclass(frozen=True)
class PilotMetrics:
    """Observed pilot metrics for one encoding candidate."""

    encoding: EncodingName
    n_qubits: int
    depth: int
    runtime_s: float
    ood_balanced_accuracy: float
    sample_efficiency_score: float | None = None
    convergence_steps: int | None = None


@dataclass(frozen=True)
class EncodingGoNoGoCriteria:
    """Selection thresholds for promoting an encoding to full sweeps."""

    max_qubits: int = 9
    max_depth: int = 140
    max_runtime_s: float = 5.0
    min_ood_balanced_accuracy: float = 0.70


@dataclass(frozen=True)
class EncodingDecision:
    """Decision record for one candidate encoding."""

    metrics: PilotMetrics
    selected: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BinaryScopeCriteria:
    """Relative stop/defer criteria for the 9-qubit binary track.

    Binary encoding is conditional scope. We keep it only if the OOD and
    sample-efficiency gains justify the extra runtime/depth cost relative to
    angle and amplitude pilots.
    """

    max_runtime_ratio_vs_angle: float = 3.0
    max_depth_ratio_vs_angle: float = 4.0
    min_accuracy_gain_vs_angle: float = 0.02
    min_sample_efficiency_gain_vs_angle: float = 0.01
    min_accuracy_vs_amplitude: float = 0.0


@dataclass(frozen=True)
class EncodingComparison:
    """Relative comparison against angle baseline for one encoding."""

    encoding: EncodingName
    n_qubits: int
    depth: int
    runtime_s: float
    ood_balanced_accuracy: float
    sample_efficiency_score: float | None
    depth_ratio_vs_angle: float | None
    runtime_ratio_vs_angle: float | None
    accuracy_delta_vs_angle: float | None
    sample_efficiency_delta_vs_angle: float | None


def _canonicalise_state(state: NimState, *, symmetry: SymmetryMode) -> tuple[int, ...]:
    """Return a symmetry-adjusted state used before encoding."""
    arr = np.asarray(state, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("state must be a 1D heap-size vector")
    if symmetry == "canonical":
        arr = np.sort(arr)
    return tuple(int(v) for v in arr)


def angle_parameters(
    state: NimState,
    *,
    M: int = 7,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Map heaps to rotation angles ``theta_i = h_i * pi / M``."""
    state_sym = _canonicalise_state(state, symmetry=symmetry)
    return np.asarray(state_sym, dtype=np.float64) * np.pi / float(M)


def build_angle_encoding_circuit(
    state: NimState,
    *,
    M: int = 7,
    symmetry: SymmetryMode = "none",
) -> QuantumCircuit:
    """Build a 3-qubit angle-encoding circuit."""
    theta = angle_parameters(state, M=M, symmetry=symmetry)
    qc = QuantumCircuit(len(theta))
    for q, angle in enumerate(theta):
        qc.ry(float(angle), q)
    return qc


def amplitude_vector(
    state: NimState,
    *,
    M: int = 7,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Build a normalised amplitude vector for pilot experiments.

    The feature vector uses normalised heap sizes and (optionally) Nim-sum:
    ``[h1/M, h2/M, h3/M, nim_sum(state)/M]``. The vector is L2-normalised and
    has dimension 4, so the circuit uses 2 qubits.
    """
    state_sym = _canonicalise_state(state, symmetry=symmetry)
    vals = [float(v) / float(M) for v in state_sym]
    if include_nim_sum:
        vals.append(float(nim_sum(state_sym)) / float(M))
    vec = np.asarray(vals, dtype=np.float64)
    if vec.size == 0:
        raise ValueError("amplitude vector cannot be empty")
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        vec = np.zeros_like(vec)
        vec[0] = 1.0
        return vec
    vec = vec / norm
    target_len = 1 << int(np.ceil(np.log2(vec.size)))
    if vec.size < target_len:
        vec = np.pad(vec, (0, target_len - vec.size))
    return vec


def build_amplitude_encoding_circuit(
    state: NimState,
    *,
    M: int = 7,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> QuantumCircuit:
    """Build amplitude-encoding pilot circuit (2 qubits for 4 amplitudes)."""
    vec = amplitude_vector(
        state,
        M=M,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,
    )
    n_qubits = int(np.log2(vec.size))
    if 2**n_qubits != vec.size:
        raise ValueError("Amplitude vector length must be a power of two")
    qc = QuantumCircuit(n_qubits)
    qc.initialize(vec, list(range(n_qubits)))
    return qc


def binary_bits(
    state: NimState,
    *,
    bits_per_heap: int = 3,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Encode each heap as fixed-width little-endian bits."""
    state_sym = _canonicalise_state(state, symmetry=symmetry)
    bits: list[int] = []
    for heap in state_sym:
        if heap < 0:
            raise ValueError("heap sizes must be non-negative")
        if heap >= 2**bits_per_heap:
            raise ValueError(
                f"heap value {heap} exceeds representable range "
                f"[0, {2**bits_per_heap - 1}]"
            )
        for bit in range(bits_per_heap):
            bits.append((heap >> bit) & 1)
    return np.asarray(bits, dtype=np.int8)


def build_binary_encoding_circuit(
    state: NimState,
    *,
    bits_per_heap: int = 3,
    symmetry: SymmetryMode = "none",
) -> QuantumCircuit:
    """Build binary encoding with Nim-sum-aligned bit-position entanglement.

    Qubits are ordered as:
    ``[h1_b0, h1_b1, h1_b2, h2_b0, ..., h3_b2]``.

    For each bit position, the three corresponding heap qubits are entangled
    with CZ gates to mirror the bitwise XOR structure in Nim-sum.
    """
    n_heaps = len(_canonicalise_state(state, symmetry="none"))
    if n_heaps != 3:
        raise ValueError("binary encoding currently expects exactly 3 heaps")
    bits = binary_bits(state, bits_per_heap=bits_per_heap, symmetry=symmetry)
    n_qubits = n_heaps * bits_per_heap
    qc = QuantumCircuit(n_qubits)

    for q, bit in enumerate(bits):
        if int(bit) == 1:
            qc.x(q)

    for bit_pos in range(bits_per_heap):
        q0 = bit_pos
        q1 = bits_per_heap + bit_pos
        q2 = 2 * bits_per_heap + bit_pos
        qc.cz(q0, q1)
        qc.cz(q1, q2)
        if symmetry == "equivariant":
            # Add symmetric closure so each heap pair is treated equally.
            qc.cz(q0, q2)
    return qc


def build_iqp_parity_feature_map(
    state: NimState,
    *,
    M: int = 7,
    reps: int = 2,
    symmetry: SymmetryMode = "none",
) -> QuantumCircuit:
    """Build an IQP-style parity feature map on 3 qubits.

    Pattern per repetition: ``H -> RZ(theta_i) -> RZZ(theta_i * theta_j)``.
    Pairwise interactions induce parity-sensitive phase correlations aligned
    with Nim-sum structure.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    theta = angle_parameters(state, M=M, symmetry=symmetry)
    n_qubits = len(theta)
    if n_qubits != 3:
        raise ValueError("IQP parity map currently expects exactly 3 heaps")
    qc = QuantumCircuit(n_qubits)
    for _ in range(reps):
        for q in range(n_qubits):
            qc.h(q)
        for q in range(n_qubits):
            qc.rz(float(theta[q]), q)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.rzz(float(theta[i] * theta[j]), i, j)
    return qc


def evaluate_go_no_go(
    metrics: PilotMetrics,
    criteria: EncodingGoNoGoCriteria | None = None,
) -> EncodingDecision:
    """Evaluate one encoding candidate against explicit go/no-go criteria."""
    crit = criteria or EncodingGoNoGoCriteria()
    reasons: list[str] = []
    selected = True

    if metrics.n_qubits > crit.max_qubits:
        selected = False
        reasons.append(f"qubits {metrics.n_qubits} > {crit.max_qubits}")
    if metrics.depth > crit.max_depth:
        selected = False
        reasons.append(f"depth {metrics.depth} > {crit.max_depth}")
    if metrics.runtime_s > crit.max_runtime_s:
        selected = False
        reasons.append(f"runtime {metrics.runtime_s:.3f}s > {crit.max_runtime_s:.3f}s")
    if metrics.ood_balanced_accuracy < crit.min_ood_balanced_accuracy:
        selected = False
        reasons.append(
            "OOD balanced accuracy "
            f"{metrics.ood_balanced_accuracy:.3f} < {crit.min_ood_balanced_accuracy:.3f}"
        )
    if selected:
        reasons.append("meets all go/no-go thresholds")
    return EncodingDecision(metrics=metrics, selected=selected, reasons=tuple(reasons))


def build_encoding_circuit(
    encoding: EncodingName,
    state: NimState,
    *,
    M: int = 7,
    bits_per_heap: int = 3,
    iqp_reps: int = 2,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> QuantumCircuit:
    """Dispatch helper to build any staged encoding circuit by name."""
    if encoding == "angle":
        return build_angle_encoding_circuit(state, M=M, symmetry=symmetry)
    if encoding == "amplitude":
        return build_amplitude_encoding_circuit(
            state,
            M=M,
            include_nim_sum=include_nim_sum,
            symmetry=symmetry,
        )
    if encoding == "binary":
        return build_binary_encoding_circuit(
            state,
            bits_per_heap=bits_per_heap,
            symmetry=symmetry,
        )
    return build_iqp_parity_feature_map(
        state,
        M=M,
        reps=iqp_reps,
        symmetry=symmetry,
    )


def pilot_metrics_from_observation(
    encoding: EncodingName,
    state: NimState,
    *,
    runtime_s: float,
    ood_balanced_accuracy: float,
    sample_efficiency_score: float | None = None,
    convergence_steps: int | None = None,
    M: int = 7,
    bits_per_heap: int = 3,
    iqp_reps: int = 2,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> PilotMetrics:
    """Create ``PilotMetrics`` from observed runtime/quality and circuit stats."""
    qc = build_encoding_circuit(
        encoding,
        state,
        M=M,
        bits_per_heap=bits_per_heap,
        iqp_reps=iqp_reps,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,
    )
    return PilotMetrics(
        encoding=encoding,
        n_qubits=qc.num_qubits,
        depth=qc.depth(),
        runtime_s=runtime_s,
        ood_balanced_accuracy=ood_balanced_accuracy,
        sample_efficiency_score=sample_efficiency_score,
        convergence_steps=convergence_steps,
    )


def select_encodings_for_sweeps(
    pilot_metrics: list[PilotMetrics],
    criteria: EncodingGoNoGoCriteria | None = None,
) -> list[EncodingDecision]:
    """Return go/no-go decisions for all pilot candidates."""
    return [evaluate_go_no_go(m, criteria) for m in pilot_metrics]


def compare_encoding_pilots(
    pilot_metrics: list[PilotMetrics],
    *,
    reference_encoding: EncodingName = "angle",
) -> list[EncodingComparison]:
    """Compare candidate encodings to a reference pilot.

    The comparison reports absolute metrics and relative ratios/deltas for
    depth, runtime, OOD balanced accuracy, and sample-efficiency score.
    """
    ref = next((m for m in pilot_metrics if m.encoding == reference_encoding), None)
    if ref is None:
        raise ValueError(f"reference encoding '{reference_encoding}' not found")
    if ref.depth <= 0:
        raise ValueError("reference depth must be > 0")
    if ref.runtime_s <= 0.0:
        raise ValueError("reference runtime must be > 0")

    rows: list[EncodingComparison] = []
    for metrics in pilot_metrics:
        depth_ratio: float | None = metrics.depth / ref.depth
        runtime_ratio: float | None = metrics.runtime_s / ref.runtime_s
        acc_delta: float | None = metrics.ood_balanced_accuracy - ref.ood_balanced_accuracy

        sample_eff_delta: float | None = None
        if (
            metrics.sample_efficiency_score is not None
            and ref.sample_efficiency_score is not None
        ):
            sample_eff_delta = (
                metrics.sample_efficiency_score - ref.sample_efficiency_score
            )

        rows.append(
            EncodingComparison(
                encoding=metrics.encoding,
                n_qubits=metrics.n_qubits,
                depth=metrics.depth,
                runtime_s=metrics.runtime_s,
                ood_balanced_accuracy=metrics.ood_balanced_accuracy,
                sample_efficiency_score=metrics.sample_efficiency_score,
                depth_ratio_vs_angle=depth_ratio,
                runtime_ratio_vs_angle=runtime_ratio,
                accuracy_delta_vs_angle=acc_delta,
                sample_efficiency_delta_vs_angle=sample_eff_delta,
            )
        )
    return rows


def evaluate_binary_scope(
    pilot_metrics: list[PilotMetrics],
    *,
    criteria: BinaryScopeCriteria | None = None,
) -> EncodingDecision:
    """Apply conditional-scope stop/defer logic for binary encoding.

    Decision rule:
    - Binary must meet absolute go/no-go thresholds.
    - Relative to angle, binary's extra runtime/depth must be justified by
      accuracy/sample-efficiency gains.
    - Binary must not underperform amplitude on OOD balanced accuracy.
    """
    crit = criteria or BinaryScopeCriteria()
    by_name = {m.encoding: m for m in pilot_metrics}
    missing = {"angle", "amplitude", "binary"} - set(by_name)
    if missing:
        raise ValueError(
            "evaluate_binary_scope requires angle, amplitude, and binary metrics; "
            f"missing: {sorted(missing)}"
        )

    binary = by_name["binary"]
    angle = by_name["angle"]
    amplitude = by_name["amplitude"]

    absolute = evaluate_go_no_go(binary)
    selected = absolute.selected
    reasons = list(absolute.reasons)

    runtime_ratio = binary.runtime_s / angle.runtime_s if angle.runtime_s > 0 else np.inf
    depth_ratio = binary.depth / angle.depth if angle.depth > 0 else np.inf
    acc_gain = binary.ood_balanced_accuracy - angle.ood_balanced_accuracy

    if runtime_ratio > crit.max_runtime_ratio_vs_angle:
        selected = False
        reasons.append(
            "defer: binary runtime ratio vs angle "
            f"{runtime_ratio:.2f} > {crit.max_runtime_ratio_vs_angle:.2f}"
        )
    if depth_ratio > crit.max_depth_ratio_vs_angle:
        selected = False
        reasons.append(
            "defer: binary depth ratio vs angle "
            f"{depth_ratio:.2f} > {crit.max_depth_ratio_vs_angle:.2f}"
        )
    if acc_gain < crit.min_accuracy_gain_vs_angle:
        selected = False
        reasons.append(
            "defer: binary accuracy gain vs angle "
            f"{acc_gain:.3f} < {crit.min_accuracy_gain_vs_angle:.3f}"
        )
    if binary.ood_balanced_accuracy < amplitude.ood_balanced_accuracy + crit.min_accuracy_vs_amplitude:
        selected = False
        reasons.append(
            "defer: binary accuracy does not exceed amplitude threshold "
            f"({binary.ood_balanced_accuracy:.3f} vs {amplitude.ood_balanced_accuracy:.3f})"
        )

    if (
        binary.sample_efficiency_score is not None
        and angle.sample_efficiency_score is not None
    ):
        sample_eff_gain = (
            binary.sample_efficiency_score - angle.sample_efficiency_score
        )
        if sample_eff_gain < crit.min_sample_efficiency_gain_vs_angle:
            selected = False
            reasons.append(
                "defer: binary sample-efficiency gain vs angle "
                f"{sample_eff_gain:.3f} < {crit.min_sample_efficiency_gain_vs_angle:.3f}"
            )
    else:
        reasons.append(
            "sample-efficiency gain check skipped (missing sample_efficiency_score)"
        )

    if selected:
        reasons.append("binary track selected: cost/performance trade-off is justified")
    return EncodingDecision(metrics=binary, selected=selected, reasons=tuple(reasons))

