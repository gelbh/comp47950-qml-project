"""MLflow cache loader and per-run logger for QSVM sweeps.

Holds the encoding cache-revision tag (:data:`QSVM_ENCODING_CACHE_REVISION`)
that finished MLflow runs must match to be reused; bump it when the feature
map definition changes so stale cached metrics are not resurrected.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence, cast

from qml_project.nim.encoding import EncodingName, SymmetryMode
from qml_project.training.experiment_namespace import standard_tags
from qml_project.training.mlflow_helpers import (
    set_mlflow_tracking_uri,
    log_training_run,
    parse_mlflow_bool,
)
from qml_project.training.mlflow_schema import (
    FILTER_FINISHED_QSVM,
    MetricKey,
    ParamKey,
    PipelineValue,
)

from .kernel import KernelBackend, KernelEstimatorMode
from .model import QuantumKernelResult

QSVM_ENCODING_CACHE_REVISION: str = "3"

_parse_mlflow_bool = parse_mlflow_bool


def _load_qsvm_sweep_from_mlflow(
    experiment_name: str,
    encodings: Sequence[EncodingName],
    train_sizes: Sequence[int | str],
    seeds: Sequence[int],
    *,
    full_train_size: int,
    symmetry: SymmetryMode,
    include_nim_sum_values: Sequence[bool],
    bits_per_heap: int,
    iqp_reps: int,
    compute_win_rate: bool,
    c_svc: float,
    estimator_mode: KernelEstimatorMode,
    kernel_backend: KernelBackend,
    shots: int | None,
) -> dict[tuple[str, int, int, bool], QuantumKernelResult]:
    """Load QSVM sweep grid points from MLflow (newest run wins per key)."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return {}

    set_mlflow_tracking_uri()
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return {}

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=FILTER_FINISHED_QSVM,
        order_by=["end_time DESC"],
        max_results=10_000,
    )

    inc_wanted = {bool(x) for x in include_nim_sum_values}
    wanted: set[tuple[str, int, int, bool]] = set()
    for enc in encodings:
        for tsz in train_sizes:
            size = full_train_size if tsz == "full" else int(tsz)
            for seed in seeds:
                for inc in inc_wanted:
                    wanted.add((str(enc), size, int(seed), inc))

    cache: dict[tuple[str, int, int, bool], QuantumKernelResult] = {}

    for run in runs:
        p = run.data.params
        m = run.data.metrics
        run_estimator_mode = p.get(ParamKey.ESTIMATOR_MODE, "exact_statevector")
        run_kernel_backend = p.get(ParamKey.KERNEL_BACKEND, "manual")
        run_inc = _parse_mlflow_bool(p.get(ParamKey.INCLUDE_NIM_SUM), default=True)
        run_bits_per_heap = p.get(ParamKey.BITS_PER_HEAP, "3")
        run_iqp_reps = p.get(ParamKey.IQP_REPS, "2")
        run_c_svc = p.get(ParamKey.C_SVC, "1.0")
        if (
            p.get(ParamKey.SYMMETRY) != str(symmetry)
            or run_inc not in inc_wanted
            or str(run_bits_per_heap) != str(bits_per_heap)
            or str(run_iqp_reps) != str(iqp_reps)
            or str(run_c_svc) != str(float(c_svc))
            or str(run_estimator_mode) != str(estimator_mode)
            or str(run_kernel_backend) != str(kernel_backend)
        ):
            continue
        if estimator_mode == "shot_binomial":
            if str(p.get(ParamKey.SHOTS, "")) != str(int(shots if shots is not None else 1024)):
                continue
        if p.get(ParamKey.ENCODING_CACHE_REVISION) != QSVM_ENCODING_CACHE_REVISION:
            continue
        try:
            enc = p.get(ParamKey.ENCODING)
            train_size_int = int(p[ParamKey.TRAIN_SIZE])
            seed_int = int(p[ParamKey.SEED])
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(enc, str):
            continue
        key = (enc, train_size_int, seed_int, run_inc)
        if key not in wanted or key in cache:
            continue
        if compute_win_rate and MetricKey.WIN_RATE not in m:
            continue
        win_rate: float | None = (
            float(m[MetricKey.WIN_RATE]) if MetricKey.WIN_RATE in m else None
        )
        loaded_shots = (
            int(shots) if (estimator_mode == "shot_binomial" and shots is not None) else None
        )
        cache[key] = QuantumKernelResult(
            encoding=cast(EncodingName, enc),
            train_size=train_size_int,
            seed=seed_int,
            accuracy=float(m.get(MetricKey.ACCURACY, 0.0)),
            balanced_accuracy=float(m.get(MetricKey.BALANCED_ACCURACY, 0.0)),
            mcc=float(m.get(MetricKey.MCC, 0.0)),
            f1=float(m.get(MetricKey.F1, 0.0)),
            precision=float(m.get(MetricKey.PRECISION, 0.0)),
            recall=float(m.get(MetricKey.RECALL, 0.0)),
            train_time_s=float(m.get(MetricKey.TRAIN_TIME_S, 0.0)),
            inference_time_s=float(m.get(MetricKey.INFERENCE_TIME_S, 0.0)),
            symmetry=symmetry,
            win_rate=win_rate,
            cm=None,
            c_svc=float(c_svc),
            estimator_mode=estimator_mode,
            kernel_backend=kernel_backend,
            shots=loaded_shots,
            include_nim_sum=run_inc,
            kernel_matrix_time_s=float(m.get(MetricKey.KERNEL_MATRIX_TIME_S, 0.0)),
        )

    return cache


def _log_mlflow_qsvm(
    result: QuantumKernelResult,
    mlflow: Any,
    *,
    bits_per_heap: int,
    iqp_reps: int,
    c_svc: float,
    estimator_mode: KernelEstimatorMode,
    kernel_backend: KernelBackend,
    shots: int | None,
    run_name_prefix: str | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> None:
    """Log one QSVM run to MLflow."""
    try:
        prefix = run_name_prefix or "qsvm"
        ns = "T" if result.include_nim_sum else "F"
        run_name = f"{prefix}|{result.encoding}|n={result.train_size}|s={result.seed}|ns={ns}"
        merged: dict[str, Any] = {
            ParamKey.PIPELINE: PipelineValue.QSVM,
            ParamKey.ENCODING: result.encoding,
            ParamKey.TRAIN_SIZE: result.train_size,
            ParamKey.SEED: result.seed,
            ParamKey.SYMMETRY: result.symmetry,
            ParamKey.INCLUDE_NIM_SUM: result.include_nim_sum,
            ParamKey.ENCODING_CACHE_REVISION: QSVM_ENCODING_CACHE_REVISION,
            ParamKey.BITS_PER_HEAP: bits_per_heap,
            ParamKey.IQP_REPS: iqp_reps,
            ParamKey.C_SVC: float(c_svc),
            ParamKey.ESTIMATOR_MODE: estimator_mode,
            ParamKey.KERNEL_BACKEND: kernel_backend,
        }
        if "|" in prefix:
            stage, variant_rest = prefix.split("|", 1)
            merged.setdefault(ParamKey.MLFLOW_RUN_PREFIX_STAGE, stage)
            merged.setdefault(ParamKey.VARIANT_ID, variant_rest)
        if extra_params:
            merged.update(dict(extra_params))
        if shots is not None:
            merged[ParamKey.SHOTS] = int(shots)
        vid = merged.get(ParamKey.VARIANT_ID)
        tags = standard_tags(
            pipeline="qsvm",
            stage="tuning",
            encoding=str(result.encoding),
            symmetry=str(result.symmetry),
            train_size=result.train_size,
            seed=result.seed,
            c_svc=float(c_svc),
            include_nim_sum=result.include_nim_sum,
            variant_id=str(vid) if vid not in (None, "") else None,
            encoding_cache_revision=QSVM_ENCODING_CACHE_REVISION,
        )
        metrics = {
            MetricKey.ACCURACY: result.accuracy,
            MetricKey.BALANCED_ACCURACY: result.balanced_accuracy,
            MetricKey.MCC: result.mcc,
            MetricKey.F1: result.f1,
            MetricKey.PRECISION: result.precision,
            MetricKey.RECALL: result.recall,
            MetricKey.TRAIN_TIME_S: result.train_time_s,
            MetricKey.KERNEL_MATRIX_TIME_S: result.kernel_matrix_time_s,
            MetricKey.INFERENCE_TIME_S: result.inference_time_s,
        }
        if result.win_rate is not None:
            metrics[MetricKey.WIN_RATE] = result.win_rate
        log_training_run(
            mlflow,
            run_name=run_name,
            params=merged,
            metrics=metrics,
            tags=tags,
        )
    except Exception as exc:
        warnings.warn(f"MLflow logging failed: {exc}", stacklevel=2)


__all__ = [
    "QSVM_ENCODING_CACHE_REVISION",
]
