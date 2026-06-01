"""Per-pipeline device readiness and validation bundle (notebook §8.6, Sections 10–11)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from qml_project.training.selection import Winner


def build_device_readiness_bundle(
    quantum_winners: Mapping[str, Winner],
    quantum_winner_rows_by_pipeline: Mapping[str, pd.DataFrame],
    shots_table_by_pipeline: Mapping[str, pd.DataFrame],
    mitigation_summary_by_pipeline: Mapping[str, pd.DataFrame],
    winner_cost_summary_by_pipeline: Mapping[str, pd.DataFrame],
    overall_winner: Winner,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    """Build go/no-go readiness rows plus a per-pipeline validation bundle.

    Returns
    -------
    device_readiness_by_pipeline
        One plain dict per pipeline (winner ids, row counts, accuracy, cost).
    device_readiness
        Readiness dict for ``overall_winner.pipeline`` (empty dict if missing).
    validation_by_pipeline
        Per pipeline: readiness row, shots table, mitigation summary, cost summary.
    """
    device_readiness_by_pipeline: dict[str, dict[str, Any]] = {}
    for pipeline, w in quantum_winners.items():
        rows = quantum_winner_rows_by_pipeline[pipeline]
        shots_tbl = shots_table_by_pipeline.get(pipeline, pd.DataFrame())
        device_readiness_by_pipeline[pipeline] = {
            "winner_pipeline": w.pipeline,
            "winner_config_id": w.config_id,
            "winner_encoding": w.encoding,
            "winner_train_size_used": w.train_size_used,
            "winner_n_rows": int(len(rows)),
            "shots_to_target_rows": int(len(shots_tbl)),
            "mean_balanced_accuracy": float(w.mean_accuracy),
            "std_balanced_accuracy": float(w.std_accuracy),
            "mean_cost_s": w.mean_cost,
        }

    device_readiness = device_readiness_by_pipeline.get(
        overall_winner.pipeline, {}
    )
    validation_by_pipeline: dict[str, dict[str, Any]] = {
        pl: {
            "readiness": row,
            "shots_table": shots_table_by_pipeline.get(pl, pd.DataFrame()),
            "mitigation_summary": mitigation_summary_by_pipeline.get(
                pl, pd.DataFrame()
            ),
            "cost_summary": winner_cost_summary_by_pipeline.get(
                pl, pd.DataFrame()
            ),
        }
        for pl, row in device_readiness_by_pipeline.items()
    }
    return device_readiness_by_pipeline, device_readiness, validation_by_pipeline
