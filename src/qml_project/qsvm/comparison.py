"""Unified comparison table across QSVM, classical baselines, and simulated VQC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qml_project.training.stats import _grouped_bootstrap_summary

from .model import QuantumKernelSweepResults


def build_kernel_pipeline_comparison(
    qsvm_results: QuantumKernelSweepResults,
    *,
    classical_df: pd.DataFrame | None = None,
    vqc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a unified comparison table: classical vs QSVM vs VQC.

    Parameters
    ----------
    qsvm_results
        Results from :func:`run_quantum_kernel_sweep`.
    classical_df
        Optional DataFrame from ``SweepResults.to_dataframe()``.
        Should include at least ``model``, ``train_size``, ``seed``,
        ``balanced_accuracy``, and ``win_rate``. When ``feature_set`` is present,
        ``SVM (RBF)`` rows are labelled ``SVM (RBF) feature=raw`` vs
        ``SVM (RBF) feature=parity`` so engineered classical features are not
        conflated with raw-input comparisons.
    vqc_df
        Optional DataFrame from ``SimulatedVQCSweepResults.to_dataframe()``.
        Should include ``train_size``, ``seed``, ``balanced_accuracy``, and
        ``win_rate``.
    """
    frames: list[pd.DataFrame] = []

    qdf = qsvm_results.to_dataframe().copy()
    if not qdf.empty:
        qdf["pipeline"] = "QSVM (Quantum Kernel)"
        enc = qdf["encoding"].astype(str)
        is_amp = enc == "amplitude"
        is_ang = enc == "angle"
        is_bin = enc == "binary"
        inc = qdf["include_nim_sum"].astype(bool)
        qdf["model"] = np.where(
            is_amp & inc,
            "amplitude (+nim-sum in state)",
            np.where(
                is_amp,
                "amplitude (heap-only)",
                np.where(
                    is_ang & inc,
                    "angle (+nim-sum)",
                    np.where(
                        is_ang,
                        "angle (heap-only)",
                        np.where(
                            is_bin & inc,
                            "binary (+nim-sum register)",
                            np.where(is_bin, "binary (heap bits only)", enc),
                        ),
                    ),
                ),
            ),
        )
        qcols = [
            "pipeline",
            "model",
            "train_size",
            "seed",
            "balanced_accuracy",
            "win_rate",
            "train_time_s",
            "kernel_matrix_time_s",
            "inference_time_s",
        ]
        qcols = [c for c in qcols if c in qdf.columns]
        qdf_subset = qdf.loc[:, qcols]
        frames.append(pd.DataFrame(qdf_subset))

    if classical_df is not None and not classical_df.empty:
        cdf = classical_df.copy()
        cdf = cdf[cdf["model"].isin(["SVM (RBF)", "SVM (Angle Kernel)"])].copy()

        def _classical_pipeline_name(row: pd.Series) -> str:
            m = str(row["model"])
            if m == "SVM (RBF)" and "feature_set" in row.index:
                return f"SVM (RBF) feature={row['feature_set']}"
            return m

        cdf["pipeline"] = cdf.apply(_classical_pipeline_name, axis=1)
        cdf["model"] = cdf["pipeline"].astype(str)
        ccols = [
            "pipeline",
            "model",
            "train_size",
            "seed",
            "balanced_accuracy",
            "win_rate",
            "train_time_s",
            "inference_time_s",
        ]
        ccols = [c for c in ccols if c in cdf.columns]
        cdf_subset = cdf.loc[:, ccols]
        frames.append(pd.DataFrame(cdf_subset))

    if vqc_df is not None and not vqc_df.empty:
        sdf = vqc_df.copy()
        sdf["pipeline"] = "VQC (Simulated)"
        if "ansatz" in sdf.columns:
            sdf["model"] = sdf["ansatz"].astype(str)
        else:
            sdf["model"] = "vqc"
        sdf_subset = sdf.loc[
            :, ["pipeline", "model", "train_size", "seed", "balanced_accuracy", "win_rate"]
        ]
        frames.append(pd.DataFrame(sdf_subset))

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    metric_cols: list[str] = ["balanced_accuracy", "win_rate"]
    for c in ("train_time_s", "kernel_matrix_time_s", "inference_time_s"):
        if c in merged.columns:
            metric_cols.append(c)
    grouped = _grouped_bootstrap_summary(
        merged,
        ("pipeline", "model", "train_size"),
        tuple(metric_cols),
    )
    return grouped.sort_values(["train_size", "pipeline", "model"]).reset_index(drop=True)


__all__ = ["build_kernel_pipeline_comparison"]
