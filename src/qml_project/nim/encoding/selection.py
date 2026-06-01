"""Dual-pipeline (QSVM + VQC) encoding gating and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .circuits import EncodingName
from .pilots import BinaryScopeCriteria


@dataclass(frozen=True)
class DualPipelineEncodingMetrics:
    """Per-encoding metrics from both QSVM and VQC pipelines."""

    encoding: EncodingName
    n_qubits: int
    depth: int
    qsvm_runtime_s: float
    qsvm_ood_balanced_accuracy: float
    qsvm_sample_efficiency_score: float | None = None
    vqc_runtime_s: float = 0.0
    vqc_ood_balanced_accuracy: float = 0.0
    vqc_sample_efficiency_score: float | None = None


@dataclass(frozen=True)
class DualPipelineGateCriteria:
    """Selection thresholds and score weights for dual-pipeline gating."""

    max_qubits: int = 9
    max_depth: int = 140
    min_qsvm_ood_balanced_accuracy: float = 0.70
    min_vqc_ood_balanced_accuracy: float = 0.70
    min_qsvm_sample_efficiency_score: float | None = None
    min_vqc_sample_efficiency_score: float | None = None
    max_qsvm_runtime_s: float | None = None
    max_vqc_runtime_s: float | None = None
    weight_qsvm_accuracy: float = 0.5
    weight_vqc_accuracy: float = 0.5
    penalty_qsvm_runtime: float = 0.05
    penalty_vqc_runtime: float = 0.05
    binary_scope_criteria: BinaryScopeCriteria | None = None


@dataclass(frozen=True)
class DualPipelineEncodingDecision:
    """Decision record for one encoding using joint QSVM+VQC evidence."""

    metrics: DualPipelineEncodingMetrics
    selected: bool
    unified_score: float
    reasons: tuple[str, ...]


def evaluate_dual_pipeline_go_no_go(
    metrics: DualPipelineEncodingMetrics,
    criteria: DualPipelineGateCriteria | None = None,
    all_metrics: Mapping[EncodingName, DualPipelineEncodingMetrics] | None = None,
) -> DualPipelineEncodingDecision:
    """Evaluate one encoding candidate against dual-pipeline thresholds."""
    crit = criteria or DualPipelineGateCriteria()
    reasons: list[str] = []
    selected = True

    if metrics.n_qubits > crit.max_qubits:
        selected = False
        reasons.append(f"qubits {metrics.n_qubits} > {crit.max_qubits}")
    if metrics.depth > crit.max_depth:
        selected = False
        reasons.append(f"depth {metrics.depth} > {crit.max_depth}")
    if metrics.qsvm_ood_balanced_accuracy < crit.min_qsvm_ood_balanced_accuracy:
        selected = False
        reasons.append(
            "QSVM OOD balanced accuracy "
            f"{metrics.qsvm_ood_balanced_accuracy:.3f} < {crit.min_qsvm_ood_balanced_accuracy:.3f}"
        )
    if metrics.vqc_ood_balanced_accuracy < crit.min_vqc_ood_balanced_accuracy:
        selected = False
        reasons.append(
            "VQC OOD balanced accuracy "
            f"{metrics.vqc_ood_balanced_accuracy:.3f} < {crit.min_vqc_ood_balanced_accuracy:.3f}"
        )
    if (
        crit.min_qsvm_sample_efficiency_score is not None
        and metrics.qsvm_sample_efficiency_score is not None
        and metrics.qsvm_sample_efficiency_score < crit.min_qsvm_sample_efficiency_score
    ):
        selected = False
        reasons.append(
            "QSVM sample-efficiency score "
            f"{metrics.qsvm_sample_efficiency_score:.3f} < {crit.min_qsvm_sample_efficiency_score:.3f}"
        )
    if (
        crit.min_vqc_sample_efficiency_score is not None
        and metrics.vqc_sample_efficiency_score is not None
        and metrics.vqc_sample_efficiency_score < crit.min_vqc_sample_efficiency_score
    ):
        selected = False
        reasons.append(
            "VQC sample-efficiency score "
            f"{metrics.vqc_sample_efficiency_score:.3f} < {crit.min_vqc_sample_efficiency_score:.3f}"
        )
    if crit.max_qsvm_runtime_s is not None and metrics.qsvm_runtime_s > crit.max_qsvm_runtime_s:
        selected = False
        reasons.append(
            "QSVM runtime "
            f"{metrics.qsvm_runtime_s:.3f}s > {crit.max_qsvm_runtime_s:.3f}s"
        )
    if crit.max_vqc_runtime_s is not None and metrics.vqc_runtime_s > crit.max_vqc_runtime_s:
        selected = False
        reasons.append(
            "VQC runtime "
            f"{metrics.vqc_runtime_s:.3f}s > {crit.max_vqc_runtime_s:.3f}s"
        )

    unified_score = (
        crit.weight_qsvm_accuracy * metrics.qsvm_ood_balanced_accuracy
        + crit.weight_vqc_accuracy * metrics.vqc_ood_balanced_accuracy
        - crit.penalty_qsvm_runtime * metrics.qsvm_runtime_s
        - crit.penalty_vqc_runtime * metrics.vqc_runtime_s
    )

    if metrics.encoding == "binary":
        bin_crit = crit.binary_scope_criteria or BinaryScopeCriteria()
        if all_metrics is None:
            selected = False
            reasons.append("binary scope check unavailable (missing all_metrics context)")
        else:
            angle = all_metrics.get("angle")
            amplitude = all_metrics.get("amplitude")
            if angle is None or amplitude is None:
                selected = False
                reasons.append(
                    "binary scope check requires angle and amplitude metrics in dual set"
                )
            else:
                runtime_ratio = (
                    metrics.qsvm_runtime_s / angle.qsvm_runtime_s
                    if angle.qsvm_runtime_s > 0
                    else np.inf
                )
                depth_ratio = metrics.depth / angle.depth if angle.depth > 0 else np.inf
                acc_gain = (
                    metrics.qsvm_ood_balanced_accuracy
                    - angle.qsvm_ood_balanced_accuracy
                )

                if runtime_ratio > bin_crit.max_runtime_ratio_vs_angle:
                    selected = False
                    reasons.append(
                        "binary scope fail: QSVM runtime ratio vs angle "
                        f"{runtime_ratio:.2f} > {bin_crit.max_runtime_ratio_vs_angle:.2f}"
                    )
                if depth_ratio > bin_crit.max_depth_ratio_vs_angle:
                    selected = False
                    reasons.append(
                        "binary scope fail: depth ratio vs angle "
                        f"{depth_ratio:.2f} > {bin_crit.max_depth_ratio_vs_angle:.2f}"
                    )
                if acc_gain < bin_crit.min_accuracy_gain_vs_angle:
                    selected = False
                    reasons.append(
                        "binary scope fail: QSVM accuracy gain vs angle "
                        f"{acc_gain:.3f} < {bin_crit.min_accuracy_gain_vs_angle:.3f}"
                    )
                if (
                    metrics.qsvm_ood_balanced_accuracy
                    < amplitude.qsvm_ood_balanced_accuracy
                    + bin_crit.min_accuracy_vs_amplitude
                ):
                    selected = False
                    reasons.append(
                        "binary scope fail: QSVM accuracy does not exceed amplitude threshold "
                        f"({metrics.qsvm_ood_balanced_accuracy:.3f} vs "
                        f"{amplitude.qsvm_ood_balanced_accuracy:.3f})"
                    )
                if (
                    metrics.qsvm_sample_efficiency_score is not None
                    and angle.qsvm_sample_efficiency_score is not None
                ):
                    sample_eff_gain = (
                        metrics.qsvm_sample_efficiency_score
                        - angle.qsvm_sample_efficiency_score
                    )
                    if sample_eff_gain < bin_crit.min_sample_efficiency_gain_vs_angle:
                        selected = False
                        reasons.append(
                            "binary scope fail: QSVM sample-efficiency gain vs angle "
                            f"{sample_eff_gain:.3f} < "
                            f"{bin_crit.min_sample_efficiency_gain_vs_angle:.3f}"
                        )
                else:
                    reasons.append(
                        "binary scope note: sample-efficiency gain check skipped "
                        "(missing sample-efficiency score)"
                    )
                if selected:
                    reasons.append("binary scope pass: cost/performance trade-off justified")

    if selected:
        reasons.append("meets dual-pipeline gate thresholds")
    return DualPipelineEncodingDecision(
        metrics=metrics,
        selected=selected,
        unified_score=float(unified_score),
        reasons=tuple(reasons),
    )


def select_dual_pipeline_encodings(
    dual_metrics: list[DualPipelineEncodingMetrics],
    criteria: DualPipelineGateCriteria | None = None,
) -> list[DualPipelineEncodingDecision]:
    """Evaluate dual-pipeline decisions for all encoding candidates."""
    by_encoding = {m.encoding: m for m in dual_metrics}
    return [
        evaluate_dual_pipeline_go_no_go(
            m,
            criteria,
            all_metrics=by_encoding,
        )
        for m in dual_metrics
    ]


def rank_dual_pipeline_encodings(
    decisions: list[DualPipelineEncodingDecision],
    *,
    close_gap_threshold: float = 0.02,
    max_selected: int = 2,
) -> tuple[list[EncodingName], EncodingName, float | None]:
    """Apply top-k-if-close ranking over dual-pipeline decisions.

    Returns ``(selected_encodings, primary_encoding, top_score_gap)``.
    """
    if not decisions:
        raise ValueError("decisions must not be empty")

    ranked = sorted(
        decisions,
        key=lambda d: (
            d.selected,
            d.unified_score,
            d.metrics.qsvm_ood_balanced_accuracy,
            d.metrics.vqc_ood_balanced_accuracy,
            -d.metrics.qsvm_runtime_s,
        ),
        reverse=True,
    )
    primary = ranked[0].metrics.encoding
    top_gap: float | None = None
    if len(ranked) >= 2:
        top_gap = float(ranked[0].unified_score - ranked[1].unified_score)

    selected: list[EncodingName] = [primary]
    if (
        len(ranked) >= 2
        and top_gap is not None
        and top_gap <= close_gap_threshold
        and max_selected >= 2
    ):
        selected.append(ranked[1].metrics.encoding)
    return selected, primary, top_gap
