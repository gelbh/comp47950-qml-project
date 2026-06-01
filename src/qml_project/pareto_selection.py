"""Pareto accuracy-vs-cost selection over VQC and QSVM workflow frames (§7).

Aggregates per configuration at the maximum observed ``train_size``, builds the
global Pareto front for labelling, and picks per-pipeline winners on each
pipeline's own front so a globally dominated family still yields a row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from qml_project.training.selection import Winner

DEFAULT_VQC_GROUP_COLS: tuple[str, ...] = ("config_id",)
DEFAULT_QSVM_GROUP_COLS: tuple[str, ...] = ("variant_id", "encoding", "include_nim_sum")


@dataclass(frozen=True)
class ParetoQuantumSelection:
    """Outputs of :func:`build_pareto_quantum_selection`."""

    selection_table: pd.DataFrame
    quantum_winners: dict[str, Winner]
    winner: Winner


@dataclass(frozen=True)
class QuantumWinnerArtifacts:
    """Per-pipeline winner row slices plus overall-winner mirrors.

    ``quantum_winner`` and ``quantum_winner_rows`` are the per-pipeline winner
    and row frame for the same pipeline as ``overall_winner`` passed to
    :func:`build_quantum_winner_artifacts` (redundant with indexing the dict
    when the caller already holds the overall ``Winner``).
    """

    quantum_winner_rows_by_pipeline: dict[str, pd.DataFrame]
    quantum_winner: Winner
    quantum_winner_rows: pd.DataFrame


def filter_workflow_rows_to_winner(frame: pd.DataFrame, w: Winner) -> pd.DataFrame:
    """Return rows of ``frame`` matching the winner's full composite key.

    Matches on every column in ``w.match_keys`` (e.g. ``variant_id`` +
    ``encoding`` for QSVM, ``config_id`` for VQC). Falls back to the legacy
    single-column match if ``match_keys`` is empty, so hand-built ``Winner``
    instances still work.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    if not w.match_keys:
        for col in ("config_id", "variant_id"):
            if col in frame.columns:
                return frame.loc[frame[col].astype(str) == w.config_id].copy()
        return pd.DataFrame()
    mask = pd.Series(True, index=frame.index)
    for col, val in w.match_keys.items():
        if col not in frame.columns:
            return pd.DataFrame()
        if pd.isna(val):
            mask &= frame[col].isna()
        else:
            mask &= frame[col].astype(str) == str(val)
    return frame.loc[mask].copy()


def build_quantum_winner_artifacts(
    quantum_winners: Mapping[str, Winner],
    workflow_frames_by_pipeline: Mapping[str, pd.DataFrame],
    overall_winner: Winner,
) -> QuantumWinnerArtifacts:
    """Slice each pipeline's workflow frame to that pipeline's ``Winner`` rows.

    Returns ``quantum_winner_rows_by_pipeline`` keyed by pipeline, plus the
    overall winner's ``Winner`` handle and single-pipeline row frame
    (``quantum_winner``, ``quantum_winner_rows``) for the same pipeline as
    ``overall_winner`` — redundant with indexing ``quantum_winner_rows_by_pipeline``
    when callers already hold ``overall_winner`` from :func:`build_pareto_quantum_selection`.
    """
    rows_by: dict[str, pd.DataFrame] = {}
    for pipeline, w in quantum_winners.items():
        src = workflow_frames_by_pipeline.get(pipeline)
        if src is None or src.empty:
            rows_by[pipeline] = pd.DataFrame()
            continue
        rows_by[pipeline] = filter_workflow_rows_to_winner(src, w)
    qw = quantum_winners.get(overall_winner.pipeline, overall_winner)
    qwr = rows_by.get(overall_winner.pipeline, pd.DataFrame())
    return QuantumWinnerArtifacts(
        quantum_winner_rows_by_pipeline=rows_by,
        quantum_winner=qw,
        quantum_winner_rows=qwr,
    )


def _summarise_per_config(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    pipeline_label: str,
    accuracy_col: str,
    cost_col: str,
) -> pd.DataFrame:
    """Aggregate per-group mean/std accuracy and mean cost at max train_size."""
    if df.empty:
        return pd.DataFrame()
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{pipeline_label}: grouping cols missing from frame: {missing}"
        )
    work = df.copy()
    if "train_size" in work.columns:
        work["train_size"] = pd.to_numeric(work["train_size"], errors="coerce")
        max_n = work["train_size"].max()
        work = work.loc[work["train_size"] == max_n].copy()
        train_size_used = int(max_n) if pd.notna(max_n) else None
    else:
        train_size_used = None
    if work.empty:
        return pd.DataFrame()
    work[accuracy_col] = pd.to_numeric(work[accuracy_col], errors="coerce")
    agg_dict: dict[str, Any] = {
        "mean_accuracy": (accuracy_col, "mean"),
        "std_accuracy": (accuracy_col, "std"),
        "n_runs": (accuracy_col, "count"),
    }
    if cost_col in work.columns:
        work[cost_col] = pd.to_numeric(work[cost_col], errors="coerce")
        agg_dict["mean_cost"] = (cost_col, "mean")
    grouped = (
        work.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()
    )
    grouped["pipeline"] = pipeline_label
    grouped["train_size_used"] = train_size_used
    if "mean_cost" not in grouped.columns:
        grouped["mean_cost"] = np.nan
    grouped["selection_id"] = grouped[group_cols].astype(str).agg("|".join, axis=1)
    if "encoding" not in grouped.columns and "encoding" in work.columns:
        enc_map = (
            work.groupby(group_cols, dropna=False)["encoding"]
            .first()
            .reset_index()
        )
        grouped = grouped.merge(enc_map, on=group_cols, how="left")
    elif "encoding" not in grouped.columns:
        grouped["encoding"] = None
    if "include_nim_sum" not in grouped.columns and "include_nim_sum" in work.columns:
        nim_map = (
            work.groupby(group_cols, dropna=False)["include_nim_sum"]
            .first()
            .reset_index()
        )
        grouped = grouped.merge(nim_map, on=group_cols, how="left")
    elif "include_nim_sum" not in grouped.columns:
        grouped["include_nim_sum"] = np.nan
    grouped["group_cols"] = [tuple(group_cols)] * len(grouped)
    return grouped


def _pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    """Pareto-optimal rows on (max mean_accuracy, min mean_cost); NaN cost kept."""
    if frame.empty:
        return frame
    rows = frame.sort_values(
        ["mean_accuracy", "mean_cost"], ascending=[False, True]
    ).reset_index(drop=True)
    keep: list[bool] = []
    best_cost = np.inf
    for _, r in rows.iterrows():
        cost = r["mean_cost"]
        if pd.isna(cost):
            keep.append(True)
            continue
        if cost <= best_cost:
            keep.append(True)
            best_cost = cost
        else:
            keep.append(False)
    return rows.loc[keep].reset_index(drop=True)


def _row_to_winner(row: pd.Series, *, rationale: str) -> Winner:
    """Build a ``Winner`` from a row of the selection table."""
    group_cols = [str(c) for c in (row.get("group_cols") or ())]
    match_keys: dict[str, Any] = {c: row[c] for c in group_cols if c in row.index}
    enc_val = row.get("encoding")
    encoding: str | None
    if enc_val is None or pd.isna(enc_val):
        encoding = None
    else:
        encoding = str(enc_val)
    ts_used = row.get("train_size_used")
    train_size_used = int(ts_used) if pd.notna(ts_used) else None
    return Winner(
        pipeline=str(row["pipeline"]),
        config_id=str(row["selection_id"]),
        encoding=encoding,
        mean_accuracy=float(row["mean_accuracy"]),
        std_accuracy=float(0.0 if pd.isna(row["std_accuracy"]) else row["std_accuracy"]),
        mean_cost=(
            None if pd.isna(row.get("mean_cost", np.nan)) else float(row["mean_cost"])
        ),
        train_size_used=train_size_used,
        rationale=rationale,
        match_keys=match_keys,
        row=dict(row),
    )


def build_pareto_quantum_selection(
    vqc_with_cost: pd.DataFrame,
    qsvm_with_cost: pd.DataFrame,
    *,
    accuracy_col: str = "balanced_accuracy",
    cost_col: str = "training_time_s",
    vqc_group_cols: Sequence[str] | None = None,
    qsvm_group_cols: Sequence[str] | None = None,
    preferred_pipeline: str | None = None,
) -> ParetoQuantumSelection:
    """Build ``selection_table``, per-pipeline ``quantum_winners``, and overall ``winner``.

    See Section 7.2 notebook markdown for the full policy (deploy-time
    ``train_size``, grouping keys, Pareto axes, per-pipeline vs global front).
    """
    vgc = list(vqc_group_cols or DEFAULT_VQC_GROUP_COLS)
    qsg_default = list(qsvm_group_cols or DEFAULT_QSVM_GROUP_COLS)

    sel_frames: list[pd.DataFrame] = []
    if not vqc_with_cost.empty:
        sel_frames.append(
            _summarise_per_config(
                vqc_with_cost,
                group_cols=vgc,
                pipeline_label="vqc",
                accuracy_col=accuracy_col,
                cost_col=cost_col,
            )
        )
    if not qsvm_with_cost.empty:
        qcols = [c for c in qsg_default if c in qsvm_with_cost.columns] or ["config_id"]
        sel_frames.append(
            _summarise_per_config(
                qsvm_with_cost,
                group_cols=qcols,
                pipeline_label="qsvm",
                accuracy_col=accuracy_col,
                cost_col=cost_col,
            )
        )
    if not sel_frames:
        raise ValueError("Section 07: no non-empty workflow frames available.")

    selection_table = pd.concat(sel_frames, ignore_index=True)
    sel_pareto = _pareto_front(selection_table)
    selection_table["pareto"] = selection_table.apply(
        lambda r: bool(
            (
                (sel_pareto["pipeline"] == r["pipeline"])
                & (sel_pareto["selection_id"] == r["selection_id"])
            ).any()
        ),
        axis=1,
    )

    rationale_base = "; ".join(
        [
            f"pareto-front top by mean {accuracy_col} within pipeline",
            f"stability tie-break on std({accuracy_col})",
            f"cost axis = {cost_col}",
            "train_size = max (deploy-time budget)",
        ]
    )

    quantum_winners: dict[str, Winner] = {}
    for pipeline in sorted(selection_table["pipeline"].unique()):
        pipe_rows = selection_table.loc[selection_table["pipeline"] == pipeline]
        if pipe_rows.empty:
            continue
        pipe_front = _pareto_front(pipe_rows)
        if pipe_front.empty:
            continue
        pipe_top = pipe_front.sort_values(
            ["mean_accuracy", "std_accuracy"], ascending=[False, True]
        ).iloc[0]
        quantum_winners[pipeline] = _row_to_winner(
            pipe_top, rationale=rationale_base
        )

    if not quantum_winners:
        raise ValueError("Section 07: no winners produced — selection_table is empty.")

    sel_ranking = sel_pareto.sort_values(
        ["mean_accuracy", "std_accuracy"], ascending=[False, True]
    ).reset_index(drop=True)
    top = sel_ranking.iloc[0]
    if len(sel_ranking) > 1:
        second = sel_ranking.iloc[1]
        tied = np.isclose(top["mean_accuracy"], second["mean_accuracy"]) and np.isclose(
            top["std_accuracy"], second["std_accuracy"]
        )
        if (
            tied
            and preferred_pipeline is not None
            and second["pipeline"] == preferred_pipeline
        ):
            top = second

    overall_rationale = rationale_base + (
        f"; documented pref = {preferred_pipeline}" if preferred_pipeline else ""
    ) + "; overall pareto-top across pipelines"
    winner = _row_to_winner(top, rationale=overall_rationale)

    winner_keys = {(w.pipeline, w.config_id) for w in quantum_winners.values()}
    selection_table["winner"] = selection_table.apply(
        lambda r: (r["pipeline"], r["selection_id"]) in winner_keys,
        axis=1,
    )

    return ParetoQuantumSelection(
        selection_table=selection_table,
        quantum_winners=quantum_winners,
        winner=winner,
    )


__all__ = [
    "DEFAULT_QSVM_GROUP_COLS",
    "DEFAULT_VQC_GROUP_COLS",
    "ParetoQuantumSelection",
    "QuantumWinnerArtifacts",
    "build_pareto_quantum_selection",
    "build_quantum_winner_artifacts",
    "filter_workflow_rows_to_winner",
]
