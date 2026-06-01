"""Write ``notebooks/.workflow_cache/*.parquet`` for ``apps/nim_demo`` loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from qml_project.notebook_setup import workflow_cache_path
from qml_project.training.selection import Winner


def _write_parquet(df: pd.DataFrame, stem: str) -> Path:
    path = workflow_cache_path(f"{stem}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def export_nim_demo_classical_df(classical_df: pd.DataFrame) -> Path:
    """Persist §3.8 ``classical_df`` for the Streamlit demo."""
    return _write_parquet(classical_df, "classical_df")


def export_nim_demo_vqc_workflow_df(vqc_workflow_df: pd.DataFrame) -> Path:
    """Persist Section 05 tuning frame."""
    return _write_parquet(vqc_workflow_df, "vqc_workflow_df")


def export_nim_demo_qsvm_workflow_df(qsvm_workflow_df: pd.DataFrame) -> Path:
    """Persist Section 06 tuning frame."""
    return _write_parquet(qsvm_workflow_df, "qsvm_workflow_df")


def build_quantum_winners_summary_df(
    quantum_winners: Mapping[str, Winner],
    overall_winner: Winner,
) -> pd.DataFrame:
    """One row per pipeline winner; ``overall_top`` marks the Pareto-global pick."""
    rows: list[dict[str, Any]] = []
    for _pipeline, w in quantum_winners.items():
        base = dict(w.row)
        base["pipeline"] = w.pipeline
        base["config_id"] = w.config_id
        base["mean_accuracy"] = float(w.mean_accuracy)
        base["std_accuracy"] = float(w.std_accuracy)
        if w.train_size_used is not None:
            base["train_size_used"] = int(w.train_size_used)
        if w.encoding is not None:
            base.setdefault("encoding", w.encoding)
        base["overall_top"] = bool(
            w.pipeline == overall_winner.pipeline
            and str(w.config_id) == str(overall_winner.config_id)
        )
        rows.append(base)
    return pd.DataFrame(rows)


def export_nim_demo_quantum_selection_parquets(
    *,
    selection_table: pd.DataFrame,
    quantum_winner_rows_by_pipeline: Mapping[str, pd.DataFrame],
    quantum_winners: Mapping[str, Winner],
    overall_winner: Winner,
) -> dict[str, Path]:
    """Persist §7 selection artefacts (``load_summary_dataframes`` contract)."""
    out: dict[str, Path] = {}
    if selection_table is not None and not selection_table.empty:
        _sel = selection_table.drop(columns=["group_cols"], errors="ignore")
        out["selection_table"] = _write_parquet(_sel, "selection_table")
    for pl in ("vqc", "qsvm"):
        df = quantum_winner_rows_by_pipeline.get(pl)
        if isinstance(df, pd.DataFrame) and not df.empty:
            out[f"quantum_winner_rows_{pl}"] = _write_parquet(
                df, f"quantum_winner_rows_{pl}"
            )
    summary = build_quantum_winners_summary_df(quantum_winners, overall_winner)
    if not summary.empty:
        out["quantum_winners_summary"] = _write_parquet(
            summary, "quantum_winners_summary"
        )
    return out


__all__ = [
    "build_quantum_winners_summary_df",
    "export_nim_demo_classical_df",
    "export_nim_demo_qsvm_workflow_df",
    "export_nim_demo_quantum_selection_parquets",
    "export_nim_demo_vqc_workflow_df",
]
