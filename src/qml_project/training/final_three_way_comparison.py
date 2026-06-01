"""Assemble classical × simulated-quantum × device comparison frames (notebook §11)."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from qml_project.training.cost_metrics import (
    add_cost_metric_contract_columns,
    summarize_cost_performance,
)
from qml_project.training.stats import paired_cross_pipeline_stat_tests


@dataclass(frozen=True)
class FinalThreeWayComparisonBundle:
    """Outputs of :func:`build_final_three_way_comparison`."""

    classical_raw_best_rows: pd.DataFrame
    classical_raw_best_info: dict[str, Any] | None
    classical_parity_best_rows: pd.DataFrame
    classical_parity_best_info: dict[str, Any] | None
    comparison_long_df: pd.DataFrame
    comparison_cost_summary: pd.DataFrame
    comparison_stat_tests: pd.DataFrame
    comparison_train_size_summary: pd.DataFrame


_NS_TF_SUFFIX = re.compile(r"\|ns=([TF])\s*$")


def vqc_heap_only_config_id(winner_config_id: str) -> str:
    """Return the VQC ``config_id`` with heap-only (Nim-sum channel off) suffix.

    Grid ids use ``|ns=T`` / ``|ns=F`` (:func:`qml_project.vqc_workflow.build_vqc_tuning_config_grid`).
    If ``winner_config_id`` already ends in ``|ns=F``, it is returned unchanged.
    If it ends in ``|ns=T``, that suffix is replaced by ``|ns=F``. Otherwise the
    string is returned unchanged (legacy ids without an ``|ns=`` marker).
    """
    cid = str(winner_config_id)
    m = _NS_TF_SUFFIX.search(cid)
    if not m:
        return cid
    if m.group(1) == "F":
        return cid
    return cid[: m.start()] + "|ns=F"


def build_paired_heap_only_quantum_workflow_rows(
    quantum_winners: Mapping[str, Any],
    *,
    qsvm_workflow_df: pd.DataFrame | None,
    vqc_workflow_df: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Slice QSVM/VQC workflow frames to heap-only configs paired to §7 winners.

    Used in the notebook to extend ``quantum_winner_rows_by_pipeline`` before
    :func:`build_final_three_way_comparison`, producing pipelines
    ``sim_quantum_qsvm_heap_only`` and ``sim_quantum_vqc_heap_only``.

    **QSVM:** same ``(variant_id, encoding)`` as the ``qsvm`` winner, with
    ``include_nim_sum`` false. If the winner already has heap-only encoding,
    this slice matches the winner rows (duplicate series under a second label).

    **VQC:** :func:`vqc_heap_only_config_id` applied to the ``vqc`` winner's
    ``config_id``; if the winner already uses ``|ns=F``, the slice matches the
    winner rows.
    """
    out: dict[str, pd.DataFrame] = {
        "qsvm_heap_only": pd.DataFrame(),
        "vqc_heap_only": pd.DataFrame(),
    }

    if qsvm_workflow_df is not None and not qsvm_workflow_df.empty:
        w_q = quantum_winners.get("qsvm")
        if w_q is not None:
            mk = getattr(w_q, "match_keys", None) or {}
            vid, enc = mk.get("variant_id"), mk.get("encoding")
            if vid is not None and enc is not None and "include_nim_sum" in qsvm_workflow_df.columns:
                sub = qsvm_workflow_df.loc[
                    (qsvm_workflow_df["variant_id"].astype(str) == str(vid))
                    & (qsvm_workflow_df["encoding"].astype(str) == str(enc))
                    & (qsvm_workflow_df["include_nim_sum"] == False)  # noqa: E712
                ].copy()
                out["qsvm_heap_only"] = sub

    if vqc_workflow_df is not None and not vqc_workflow_df.empty:
        w_v = quantum_winners.get("vqc")
        if w_v is not None and "config_id" in vqc_workflow_df.columns:
            cid_heap = vqc_heap_only_config_id(str(w_v.config_id))
            sub = vqc_workflow_df.loc[
                vqc_workflow_df["config_id"].astype(str) == cid_heap
            ].copy()
            out["vqc_heap_only"] = sub

    return out


def select_classical_winner_rows(
    df: pd.DataFrame,
    *,
    feature_set: str,
    metric: str = "balanced_accuracy",
    group_cols: tuple[str, ...] = ("model", "symmetry"),
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Per-seed rows of the best (model, symmetry) within ``feature_set`` for Section 11.

    Winner = (model, symmetry) with highest mean ``metric`` across all ``train_size``.
    """
    if df is None or df.empty or "feature_set" not in df.columns:
        return pd.DataFrame(), None
    sub = df.loc[df["feature_set"].astype(str) == feature_set].copy()
    if sub.empty or metric not in sub.columns:
        return pd.DataFrame(), None
    missing = [c for c in group_cols if c not in sub.columns]
    if missing:
        return pd.DataFrame(), None
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    per_config = (
        sub.groupby(list(group_cols), dropna=False)[metric]
        .mean()
        .reset_index()
        .sort_values(metric, ascending=False)
    )
    if per_config.empty or not np.isfinite(per_config[metric].iloc[0]):
        return pd.DataFrame(), None
    winner = per_config.iloc[0]
    mask = np.ones(len(sub), dtype=bool)
    for c in group_cols:
        mask &= sub[c].astype(object) == winner[c]
    winner_rows = sub.loc[mask].copy()
    winner_info: dict[str, Any] = {
        "feature_set": feature_set,
        **{c: winner[c] for c in group_cols},
        "selection_mean_balanced_accuracy": float(winner[metric]),
        "n_rows": int(len(winner_rows)),
    }
    return winner_rows, winner_info


def cmp_coerce_to_long(
    df: pd.DataFrame | None,
    *,
    pipeline_label: str,
    cmp_accuracy_col: str = "balanced_accuracy",
    cmp_win_rate_col: str = "win_rate",
) -> pd.DataFrame:
    """Normalize a pipeline frame to the Section 11 comparison schema."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["pipeline"] = pipeline_label
    if cmp_accuracy_col != "balanced_accuracy" and "balanced_accuracy" not in out.columns:
        out["balanced_accuracy"] = pd.to_numeric(out[cmp_accuracy_col], errors="coerce")
    if (
        cmp_win_rate_col
        and cmp_win_rate_col in out.columns
        and "win_rate" not in out.columns
    ):
        out["win_rate"] = pd.to_numeric(out[cmp_win_rate_col], errors="coerce")
    if "win_rate" not in out.columns:
        out["win_rate"] = np.nan
    return out


def build_final_three_way_comparison(
    classical_df: pd.DataFrame,
    *,
    quantum_winner_rows_by_pipeline: Mapping[str, pd.DataFrame],
    device_results_by_pipeline: Mapping[str, pd.DataFrame] | None,
    device_df: pd.DataFrame,
    n_inference_samples_test_set: int,
    cmp_accuracy_col: str = "balanced_accuracy",
    cmp_win_rate_col: str = "win_rate",
    cmp_metrics_for_stat_tests: tuple[str, ...] = ("balanced_accuracy", "win_rate"),
    cmp_alpha: float = 0.05,
) -> FinalThreeWayComparisonBundle:
    """Build long comparison frame, cost summary, Wilcoxon table, and train-size summary."""
    classical_for_cmp = classical_df.loc[classical_df["sub_study"] == "main"].copy()

    classical_raw_best_rows, classical_raw_best_info = select_classical_winner_rows(
        classical_for_cmp, feature_set="raw"
    )
    classical_parity_best_rows, classical_parity_best_info = select_classical_winner_rows(
        classical_for_cmp, feature_set="parity"
    )

    cmp_frames: list[pd.DataFrame] = [
        cmp_coerce_to_long(
            classical_raw_best_rows,
            pipeline_label="classical_raw_best",
            cmp_accuracy_col=cmp_accuracy_col,
            cmp_win_rate_col=cmp_win_rate_col,
        ),
        cmp_coerce_to_long(
            classical_parity_best_rows,
            pipeline_label="classical_parity_best",
            cmp_accuracy_col=cmp_accuracy_col,
            cmp_win_rate_col=cmp_win_rate_col,
        ),
        cmp_coerce_to_long(
            classical_for_cmp,
            pipeline_label="classical_pool",
            cmp_accuracy_col=cmp_accuracy_col,
            cmp_win_rate_col=cmp_win_rate_col,
        ),
    ]
    for _pipeline, _rows in quantum_winner_rows_by_pipeline.items():
        cmp_frames.append(
            cmp_coerce_to_long(
                _rows,
                pipeline_label=f"sim_quantum_{_pipeline}",
                cmp_accuracy_col=cmp_accuracy_col,
                cmp_win_rate_col=cmp_win_rate_col,
            )
        )
    dev_map = device_results_by_pipeline or {}
    if dev_map:
        for _pipeline, _rows in dev_map.items():
            cmp_frames.append(
                cmp_coerce_to_long(
                    _rows,
                    pipeline_label=f"device_quantum_{_pipeline}",
                    cmp_accuracy_col=cmp_accuracy_col,
                    cmp_win_rate_col=cmp_win_rate_col,
                )
            )
    elif not device_df.empty:
        cmp_frames.append(
            cmp_coerce_to_long(
                device_df,
                pipeline_label="device_quantum",
                cmp_accuracy_col=cmp_accuracy_col,
                cmp_win_rate_col=cmp_win_rate_col,
            )
        )
    cmp_frames = [f for f in cmp_frames if not f.empty]

    if not cmp_frames:
        empty = pd.DataFrame()
        return FinalThreeWayComparisonBundle(
            classical_raw_best_rows=classical_raw_best_rows,
            classical_raw_best_info=classical_raw_best_info,
            classical_parity_best_rows=classical_parity_best_rows,
            classical_parity_best_info=classical_parity_best_info,
            comparison_long_df=empty,
            comparison_cost_summary=empty,
            comparison_stat_tests=empty,
            comparison_train_size_summary=empty,
        )

    comparison_long_df = pd.concat(cmp_frames, ignore_index=True, sort=False)

    cmp_cost_frames: list[pd.DataFrame] = []
    for _frame in cmp_frames:
        _pipeline_label = str(_frame["pipeline"].iloc[0])
        if _pipeline_label.startswith("device") and "n_test" in _frame.columns:
            _n_inf_samples = int(
                pd.to_numeric(_frame["n_test"], errors="coerce").dropna().iloc[0]
            )
        else:
            _n_inf_samples = int(n_inference_samples_test_set)
        cmp_cost_frames.append(
            add_cost_metric_contract_columns(
                _frame,
                pipeline=_pipeline_label,
                n_inference_samples=_n_inf_samples,
            )
        )
    cmp_cost_tagged = pd.concat(cmp_cost_frames, ignore_index=True, sort=False)
    comparison_cost_summary = summarize_cost_performance(
        cmp_cost_tagged, group_cols=("pipeline",)
    )

    cmp_train_sizes = sorted(
        pd.to_numeric(comparison_long_df["train_size"], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )

    cmp_stat_frames: list[pd.DataFrame] = []
    for _metric in cmp_metrics_for_stat_tests:
        if _metric not in comparison_long_df.columns:
            continue
        try:
            _stats_df = paired_cross_pipeline_stat_tests(
                comparison_long_df,
                metric=_metric,
                train_sizes=cmp_train_sizes,
                alpha=cmp_alpha,
            )
        except Exception:
            continue
        if not _stats_df.empty:
            _stats_df = _stats_df.copy()
            _stats_df["metric"] = _metric
            cmp_stat_frames.append(_stats_df)
    comparison_stat_tests = (
        pd.concat(cmp_stat_frames, ignore_index=True)
        if cmp_stat_frames
        else pd.DataFrame()
    )

    cmp_ts_rows: list[dict[str, float | str]] = []
    for _metric in cmp_metrics_for_stat_tests:
        if _metric not in comparison_long_df.columns:
            continue
        for (_pipeline, _size), _grp in comparison_long_df.groupby(
            ["pipeline", "train_size"], dropna=False
        ):
            _vals = pd.to_numeric(_grp[_metric], errors="coerce").dropna()
            if _vals.empty:
                continue
            cmp_ts_rows.append(
                {
                    "pipeline": str(_pipeline),
                    "train_size": float(_size) if pd.notna(_size) else np.nan,
                    "metric": _metric,
                    "mean": float(_vals.mean()),
                    "std": float(_vals.std(ddof=1)) if _vals.size > 1 else 0.0,
                    "n": int(_vals.size),
                }
            )
    comparison_train_size_summary = pd.DataFrame(cmp_ts_rows)

    return FinalThreeWayComparisonBundle(
        classical_raw_best_rows=classical_raw_best_rows,
        classical_raw_best_info=classical_raw_best_info,
        classical_parity_best_rows=classical_parity_best_rows,
        classical_parity_best_info=classical_parity_best_info,
        comparison_long_df=comparison_long_df,
        comparison_cost_summary=comparison_cost_summary,
        comparison_stat_tests=comparison_stat_tests,
        comparison_train_size_summary=comparison_train_size_summary,
    )


__all__ = [
    "FinalThreeWayComparisonBundle",
    "build_final_three_way_comparison",
    "build_paired_heap_only_quantum_workflow_rows",
    "cmp_coerce_to_long",
    "select_classical_winner_rows",
    "vqc_heap_only_config_id",
]
