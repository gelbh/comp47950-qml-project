"""Sweep device refits over train sizes and write per-anchor workflow caches."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import pandas as pd

from qml_project.notebook_setup import workflow_cache_path

from .qsvm import QSVMDevicePayload, refit_qsvm_for_device
from .vqc import refit_vqc_for_device


class _TrainTestSplitLike(Protocol):
    """Minimal split object (e.g. notebook ``exp.split`` / ``OODSplit``)."""

    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any


def _sizes_for_pipeline(
    pipeline: str,
    *,
    train_sizes_by_pipeline: Mapping[str, Sequence[int]],
    shared_fallback_sizes: Sequence[int],
) -> list[int]:
    sizes = train_sizes_by_pipeline.get(pipeline, shared_fallback_sizes)
    return sorted({int(s) for s in sizes})


def _refit_one(
    pipeline: str,
    winner_row: Mapping[str, Any],
    *,
    split: _TrainTestSplitLike,
    train_size: int,
    seed: int,
    log: Callable[[str], None],
) -> Any | None:
    try:
        if pipeline == "vqc":
            return refit_vqc_for_device(
                winner_row=winner_row,
                X_train_raw=split.X_train,
                y_train=split.y_train,
                X_test_raw=split.X_test,
                y_test=split.y_test,
                train_size=int(train_size),
                seed=int(seed),
            )
        if pipeline == "qsvm":
            return refit_qsvm_for_device(
                winner_row=winner_row,
                X_train_raw=split.X_train,
                y_train=split.y_train,
                X_test_raw=split.X_test,
                y_test=split.y_test,
                train_size=int(train_size),
                seed=int(seed),
            )
        log(f"[{pipeline}@n{train_size}] unknown pipeline — skipping refit.")
        return None
    except NotImplementedError as exc:
        log(f"[{pipeline}@n{train_size}] device refit skipped: {exc}")
        return None


def run_device_refit_sweep_and_cache(
    *,
    quantum_winners: Mapping[str, Any],
    quantum_winner_rows_by_pipeline: Mapping[str, pd.DataFrame],
    split: _TrainTestSplitLike,
    train_sizes_by_pipeline: Mapping[str, Sequence[int]],
    shared_fallback_sizes: Sequence[int],
    refit_seed: int,
    log: Callable[[str], None] | None = None,
) -> dict[str, dict[int, Any]]:
    """Refit each quantum pipeline winner at configured train sizes and pickle payloads.

    Mirrors the Section 8.5 notebook sweep: skip empty winner tables, log
    refit metrics, and write ``{pipeline}_device_payload_n{size}.pkl`` under
    the workflow cache directory.
    """
    emit = log or print
    out: dict[str, dict[int, Any]] = {}

    for pipeline, _ in quantum_winners.items():
        rows = quantum_winner_rows_by_pipeline[pipeline]
        if rows.empty:
            emit(f"[{pipeline}] no winner rows — skipping device refit.")
            continue
        row = rows.iloc[0].to_dict()
        out.setdefault(pipeline, {})
        pipeline_sizes = _sizes_for_pipeline(
            pipeline,
            train_sizes_by_pipeline=train_sizes_by_pipeline,
            shared_fallback_sizes=shared_fallback_sizes,
        )
        emit(f"[{pipeline}] refit anchors: {pipeline_sizes}")
        for size in pipeline_sizes:
            payload = _refit_one(
                pipeline,
                row,
                split=split,
                train_size=int(size),
                seed=int(refit_seed),
                log=emit,
            )
            if payload is None:
                continue

            out[pipeline][int(size)] = payload
            cache_path = workflow_cache_path(
                f"{pipeline}_device_payload_n{int(size)}.pkl"
            )
            try:
                with cache_path.open("wb") as fh:
                    pickle.dump(payload, fh)
                emit(
                    f"[{pipeline}@n{size}] device refit → "
                    f"train_size={payload.train_size_used} "
                    f"seed={payload.refit_seed} "
                    f"balanced_accuracy={payload.refit_balanced_accuracy} "
                    f"→ cached {cache_path.name}"
                )
                if isinstance(payload, QSVMDevicePayload):
                    emit(
                        f"  |SV|={payload.sv_indices.shape[0]} / "
                        f"train_size={payload.train_size_used}"
                    )
            except Exception as exc:
                emit(
                    f"[{pipeline}@n{size}] (device payload cache skipped: {exc})"
                )

    return out
