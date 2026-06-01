"""QSVM tuning workflow runner used by Section 6 of the project notebook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..kernel import KernelEstimatorMode
from ..sweep import run_quantum_kernel_sweep
from .defaults import (
    QSVM_CLASS_WEIGHT,
    QSVM_COMPUTE_WIN_RATE,
    QSVM_ENCODINGS,
    QSVM_M,
    QSVM_N_GAMES_WIN_RATE,
    QSVM_SEEDS,
    QSVM_TRAIN_SIZES,
)
from .summary import (
    _annotate_qsvm_tuning_variant_frame,
    _normalize_qsvm_variant_include_nim_sum,
    add_qsvm_encoding_label_column,
    qsvm_variant_signature,
)


def run_qsvm_tuning_workflow_dataframe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    qsvm_variants: Sequence[Mapping[str, Any]],
    *,
    train_sizes: Sequence[int | str] = QSVM_TRAIN_SIZES,
    seeds: Sequence[int] = QSVM_SEEDS,
    class_weight: str | dict[int, float] | None = QSVM_CLASS_WEIGHT,
    M: int = QSVM_M,
    compute_win_rate: bool = QSVM_COMPUTE_WIN_RATE,
    n_games_win_rate: int = QSVM_N_GAMES_WIN_RATE,
    default_encodings: Sequence[str] = QSVM_ENCODINGS,
    mlflow_experiment: str | None = None,
    use_cache: bool = True,
    verbose: bool = False,
    max_workers: int | None = None,
    tqdm_desc: str = "qsvm_tuning (all variants)",
    tqdm_position: int = 0,
) -> pd.DataFrame:
    """Run each QSVM variant dict through ``run_quantum_kernel_sweep`` and concatenate.

    Default keyword values mirror :mod:`qml_project.qsvm.tuning` constants so
    MLflow cache keys stay aligned when callers omit explicit arguments.
    """
    frames: list[pd.DataFrame] = []
    outer = tqdm(qsvm_variants, desc=tqdm_desc, position=tqdm_position)
    try:
        for variant in outer:
            encodings = tuple(variant.get("encodings", default_encodings))
            signature = qsvm_variant_signature(dict(variant))
            variant_id = str(variant.get("variant_id") or signature)
            estimator_mode = cast(
                KernelEstimatorMode,
                variant.get("estimator_mode", "exact_statevector"),
            )
            outer.set_postfix_str(variant_id)

            include_nim_sum_arg = _normalize_qsvm_variant_include_nim_sum(
                variant.get("include_nim_sum", True)
            )

            sweep_result = run_quantum_kernel_sweep(
                X_train,
                y_train,
                X_test,
                y_test,
                encodings=encodings,
                train_sizes=train_sizes,
                seeds=seeds,
                class_weight=class_weight,
                M=M,
                bits_per_heap=int(variant.get("bits_per_heap", 3)),
                iqp_reps=int(variant.get("iqp_reps", 2)),
                include_nim_sum=include_nim_sum_arg,
                symmetry=variant.get("symmetry", "none"),
                compute_win_rate=compute_win_rate,
                n_games_win_rate=n_games_win_rate,
                mlflow_experiment=mlflow_experiment,
                mlflow_run_prefix=f"qsvm_tuning|{variant_id}",
                use_cache=use_cache,
                verbose=verbose,
                max_workers=max_workers,
                c_svc=float(variant.get("c_svc", 1.0)),
                estimator_mode=estimator_mode,
                kernel_backend=variant.get("kernel_backend", "manual"),
                shots=int(variant.get("shots", 1024)),
            )
            variant_df = sweep_result.to_dataframe()
            if variant_df.empty:
                continue
            frames.append(
                _annotate_qsvm_tuning_variant_frame(
                    variant_df,
                    variant,
                    variant_id=variant_id,
                    estimator_mode=estimator_mode,
                )
            )
    finally:
        outer.close()

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return add_qsvm_encoding_label_column(combined)


__all__ = ["run_qsvm_tuning_workflow_dataframe"]
