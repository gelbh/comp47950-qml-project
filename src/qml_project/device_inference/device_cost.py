"""IBM Runtime circuit and shot budget estimates for device inference (§10.2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from qml_project.training.selection import Winner

from .qsvm import QSVMDevicePayload
from .vqc import VQCDevicePayload

DeviceRefitPayload = VQCDevicePayload | QSVMDevicePayload


def _estimate_device_cost_row(
    pipeline: str,
    w: Winner,
    n_test_samples: int,
    payload: DeviceRefitPayload | None,
    train_size: int | None,
    shots_per_circuit: int,
) -> dict[str, Any]:
    """One table row: circuits and shots for one (pipeline, refit anchor) pair."""
    shots = int(shots_per_circuit)
    sv_count: int | None = None
    refit_size: int | None = None
    if pipeline == "vqc":
        circuits = int(n_test_samples)
        if payload is not None:
            ts = payload.train_size_used
            refit_size = int(ts) if ts is not None else None
            cost_source = "payload (refit)"
        else:
            cost_source = (
                "winner.train_size_used "
                "(VQC cost is train-size-independent)"
            )
    elif pipeline == "qsvm":
        if payload is not None:
            sv_count = int(payload.sv_indices.shape[0])
            refit_size = int(payload.train_size_used)
            circuits = int(n_test_samples * sv_count)
            cost_source = "payload (refit)"
        else:
            sv_count = int(w.train_size_used or 0)
            circuits = int(n_test_samples * sv_count)
            cost_source = (
                "winner.train_size_used "
                "(upper bound — no refit payload; "
                "run Section 08 §8.5 to refit at the §8.5 train-size anchors)"
            )
    else:
        raise ValueError(f"unknown pipeline for device cost: {pipeline!r}")

    return {
        "pipeline": pipeline,
        "config_id": w.config_id,
        "encoding": w.encoding,
        "target_train_size": int(train_size) if train_size is not None else None,
        "n_test_samples": int(n_test_samples),
        "selection_train_size": w.train_size_used,
        "refit_train_size": refit_size,
        "fitted_sv": sv_count,
        "shots_per_circuit": shots,
        "circuits": circuits,
        "total_shots_upper_bound": circuits * shots,
        "cost_source": cost_source,
    }


def build_device_cost_estimates_dataframe(
    *,
    quantum_winners: Mapping[str, Winner],
    device_payloads_by_pipeline: Mapping[str, Mapping[int, DeviceRefitPayload]],
    n_test_samples: int,
    shots_per_circuit: int,
) -> pd.DataFrame:
    """Build the §10.2 cost table from winners and optional refit payloads.

    When a pipeline has no cached payloads, QSVM rows use
    ``Winner.train_size_used`` as an upper bound on ``|SV|`` (see cell
    markdown). VQC circuit count is ``n_test_samples`` either way.
    """
    rows: list[dict[str, Any]] = []
    for pipeline, w in quantum_winners.items():
        per_pipeline = device_payloads_by_pipeline.get(pipeline) or {}
        if per_pipeline:
            for size in sorted(per_pipeline):
                rows.append(
                    _estimate_device_cost_row(
                        pipeline,
                        w,
                        n_test_samples,
                        per_pipeline[size],
                        size,
                        shots_per_circuit,
                    )
                )
        else:
            rows.append(
                _estimate_device_cost_row(
                    pipeline,
                    w,
                    n_test_samples,
                    None,
                    None,
                    shots_per_circuit,
                )
            )
    return pd.DataFrame(rows)


def sum_device_cost_circuits_by_pipeline(cost_df: pd.DataFrame) -> tuple[int, int]:
    """Return ``(vqc_circuits_total, qsvm_circuits_total)`` summed over rows."""
    if cost_df.empty:
        return 0, 0
    vqc = int(cost_df.loc[cost_df["pipeline"] == "vqc", "circuits"].sum())
    qsvm = int(cost_df.loc[cost_df["pipeline"] == "qsvm", "circuits"].sum())
    return vqc, qsvm
