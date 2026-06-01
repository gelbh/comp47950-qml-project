"""QSVM tuning summary helpers (variant signatures, group/agg, encoding labels)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..kernel import KernelEstimatorMode


def qsvm_variant_signature(variant: dict) -> str:
    """Stable ``|`` separated signature of QSVM variant kwargs (cache key piece)."""
    parts: list[str] = []
    for k in (
        "symmetry",
        "c_svc",
        "estimator_mode",
        "kernel_backend",
        "shots",
        "bits_per_heap",
        "iqp_reps",
        "include_nim_sum",
    ):
        if k in variant:
            v = variant[k]
            if isinstance(v, (list, tuple)):
                parts.append(f"{k}={tuple(v)}")
            else:
                parts.append(f"{k}={v}")
    return "|".join(parts)


def qsvm_summary_group_columns(workflow_df: pd.DataFrame) -> list[str]:
    """Group keys for aggregating QSVM tuning rows (inserts ``include_nim_sum`` when present)."""
    cols = ["variant_id", "encoding", "train_size"]
    if "include_nim_sum" in workflow_df.columns:
        cols.insert(2, "include_nim_sum")
    return cols


def merge_qsvm_estimator_mode_onto_summary(
    summary: pd.DataFrame, workflow_df: pd.DataFrame
) -> pd.DataFrame:
    """Carry ``estimator_mode`` through aggregation by left-joining on ``variant_id``."""
    if summary.empty or "variant_id" not in summary.columns:
        return summary
    mode_map = workflow_df.groupby("variant_id", dropna=False)["estimator_mode"].first()
    return summary.merge(mode_map.reset_index(), on="variant_id", how="left")


def summarize_qsvm_workflow_dataframe(workflow_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run QSVM rows to mean/std over seeds (Section 6.3 long table)."""
    if workflow_df.empty:
        return pd.DataFrame()
    group_columns = qsvm_summary_group_columns(workflow_df)
    aggregation_spec: dict[str, tuple[str, str]] = {
        "balanced_accuracy_mean": ("balanced_accuracy", "mean"),
        "balanced_accuracy_std": ("balanced_accuracy", "std"),
        "mcc_mean": ("mcc", "mean"),
        "mcc_std": ("mcc", "std"),
        "win_rate_mean": ("win_rate", "mean"),
        "win_rate_std": ("win_rate", "std"),
        "train_time_s_mean": ("train_time_s", "mean"),
        "train_time_s_std": ("train_time_s", "std"),
        "n_runs": ("balanced_accuracy", "count"),
    }
    if "kernel_matrix_time_s" in workflow_df.columns:
        aggregation_spec["kernel_matrix_time_s_mean"] = ("kernel_matrix_time_s", "mean")
        aggregation_spec["kernel_matrix_time_s_std"] = ("kernel_matrix_time_s", "std")
    summary = (
        workflow_df.groupby(group_columns, dropna=False).agg(**aggregation_spec).reset_index()
    )
    return merge_qsvm_estimator_mode_onto_summary(summary, workflow_df)


def qsvm_balanced_accuracy_mean_pivot(summary: pd.DataFrame) -> pd.DataFrame | None:
    """Pivot mean balanced accuracy with train sizes as columns; ``None`` if not possible."""
    if summary.empty:
        return None
    pivot_index = [c for c in ("variant_id", "encoding", "include_nim_sum") if c in summary.columns]
    if not pivot_index:
        return None
    return summary.pivot_table(
        index=pivot_index,
        columns="train_size",
        values="balanced_accuracy_mean",
        aggfunc="first",
    )


def filter_qsvm_summary_exact_statevector(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep only ``estimator_mode == 'exact_statevector'`` rows from a tuning summary."""
    if summary.empty or "estimator_mode" not in summary.columns:
        return summary
    mask = summary["estimator_mode"].astype(str) == "exact_statevector"
    return summary.loc[mask].copy()


def add_qsvm_encoding_label_column(workflow_df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable ``encoding_label`` from ``encoding`` × ``include_nim_sum``."""
    if (
        workflow_df.empty
        or "encoding" not in workflow_df.columns
        or "include_nim_sum" not in workflow_df.columns
    ):
        return workflow_df
    out = workflow_df.copy()
    enc = out["encoding"].astype(str)
    ns = out["include_nim_sum"].astype(bool)
    out["encoding_label"] = np.select(
        [
            (enc == "amplitude") & ns,
            (enc == "amplitude") & ~ns,
            (enc == "angle") & ns,
            (enc == "angle") & ~ns,
            (enc == "binary") & ns,
            (enc == "binary") & ~ns,
        ],
        [
            "amplitude (+nim-sum)",
            "amplitude (heap-only)",
            "angle (+nim-sum)",
            "angle (heap-only)",
            "binary (+nim-sum register)",
            "binary (heap bits only)",
        ],
        default=enc.astype(str),
    )
    return out


def _normalize_qsvm_variant_include_nim_sum(raw: Any) -> bool | tuple[bool, ...]:
    """Coerce ``include_nim_sum`` (scalar or sequence) to the sweep-friendly form."""
    if isinstance(raw, (list, tuple)):
        return tuple(bool(x) for x in raw)
    return bool(raw)


def _annotate_qsvm_tuning_variant_frame(
    variant_df: pd.DataFrame,
    variant: Mapping[str, Any],
    *,
    variant_id: str,
    estimator_mode: KernelEstimatorMode,
) -> pd.DataFrame:
    """Decorate one variant's raw sweep frame with metadata columns required for MLflow keys."""
    out = variant_df.copy()
    variant_meta = {
        "variant_id": variant_id,
        "symmetry": variant.get("symmetry", "none"),
        "c_svc": float(variant.get("c_svc", 1.0)),
        "estimator_mode": estimator_mode,
        "kernel_backend": variant.get("kernel_backend", "manual"),
        "shots": int(variant.get("shots", 1024)) if estimator_mode == "shot_binomial" else None,
        "bits_per_heap": int(variant.get("bits_per_heap", 3)),
        "iqp_reps": int(variant.get("iqp_reps", 2)),
    }
    for key, val in variant_meta.items():
        if key not in out.columns:
            out[key] = val
    out["pipeline"] = "qsvm"
    out["stage"] = "tuning"
    return out


__all__ = [
    "add_qsvm_encoding_label_column",
    "filter_qsvm_summary_exact_statevector",
    "merge_qsvm_estimator_mode_onto_summary",
    "qsvm_balanced_accuracy_mean_pivot",
    "qsvm_summary_group_columns",
    "qsvm_variant_signature",
    "summarize_qsvm_workflow_dataframe",
]
