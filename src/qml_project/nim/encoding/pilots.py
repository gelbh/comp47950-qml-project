"""Single-pipeline pilot metrics, go/no-go thresholds, and binary-scope check."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qml_project.nim.game import NimState

from .circuits import EncodingName, SymmetryMode, build_encoding_circuit


@dataclass(frozen=True)
class EncodingMetrics:
    """Observed metrics for one encoding candidate."""

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

    metrics: EncodingMetrics
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


def evaluate_go_no_go(
    metrics: EncodingMetrics,
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
) -> EncodingMetrics:
    """Create ``EncodingMetrics`` from observed runtime/quality and circuit stats."""
    qc = build_encoding_circuit(
        encoding,
        state,
        M=M,
        bits_per_heap=bits_per_heap,
        iqp_reps=iqp_reps,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,
    )
    return EncodingMetrics(
        encoding=encoding,
        n_qubits=qc.num_qubits,
        depth=qc.depth(),
        runtime_s=runtime_s,
        ood_balanced_accuracy=ood_balanced_accuracy,
        sample_efficiency_score=sample_efficiency_score,
        convergence_steps=convergence_steps,
    )


def select_encodings_for_sweeps(
    pilot_metrics: list[EncodingMetrics],
    criteria: EncodingGoNoGoCriteria | None = None,
) -> list[EncodingDecision]:
    """Return go/no-go decisions for all pilot candidates."""
    return [evaluate_go_no_go(m, criteria) for m in pilot_metrics]


def compare_encoding_pilots(
    pilot_metrics: list[EncodingMetrics],
    *,
    reference_encoding: EncodingName = "angle",
) -> list[EncodingComparison]:
    """Compare candidate encodings to a reference pilot.

    Reports absolute metrics and relative ratios/deltas for depth, runtime,
    OOD balanced accuracy, and sample-efficiency score.
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
    pilot_metrics: list[EncodingMetrics],
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
