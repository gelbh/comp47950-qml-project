"""MLflow cache loader and per-run logger for the classical baseline sweep."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

from qml_project.baselines.evaluation import ClassicalResult
from qml_project.training.mlflow_helpers import set_mlflow_tracking_uri
from qml_project.training.mlflow_schema import MetricKey, ParamKey, PipelineValue


def load_classical_sweep_cache(
    experiment_name: str,
    model_names: Sequence[str],
    feature_sets: Sequence[str],
    symmetry_variants: Sequence[str],
    train_sizes: Sequence[int | str],
    seeds: Sequence[int],
    regime: str,
    *,
    full_train_size: int,
    c_svc: float = 1.0,
) -> dict[tuple[str, str, str, int, int, float], ClassicalResult]:
    """Load classical sweep results from MLflow runs (cache lookup).

    Returns a dict keyed by
    ``(model_name, feature_set, symmetry, train_size, seed, c_svc)``.
    Only includes runs that match the requested grid and regime; when multiple
    runs exist for the same params, the latest (by end_time) is used.
    """
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
        order_by=["end_time DESC"],
        max_results=10_000,
    )

    c_target = float(c_svc)
    wanted: set[tuple[str, str, str, int, int, float]] = set()
    for model in model_names:
        for fs in feature_sets:
            for sym in symmetry_variants:
                for tsz in train_sizes:
                    size = full_train_size if tsz == "full" else int(tsz)
                    for seed in seeds:
                        wanted.add((model, fs, sym, size, seed, c_target))

    cache: dict[tuple[str, str, str, int, int, float], ClassicalResult] = {}
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        params = run.data.params
        metrics = run.data.metrics
        model = params.get(ParamKey.MODEL)
        fs = params.get(ParamKey.FEATURE_SET)
        sym = params.get(ParamKey.SYMMETRY)
        train_size_str = params.get(ParamKey.TRAIN_SIZE)
        seed_str = params.get(ParamKey.SEED)
        reg = params.get(ParamKey.REGIME)
        if (
            model is None
            or fs is None
            or sym is None
            or train_size_str is None
            or seed_str is None
            or reg is None
            or reg != regime
        ):
            continue
        try:
            train_size_int = int(train_size_str)
            seed_int = int(seed_str)
            run_c_svc = float(params.get(ParamKey.C_SVC, "1.0"))
        except (TypeError, ValueError):
            continue
        if run_c_svc != c_target:
            continue
        assert isinstance(model, str) and isinstance(fs, str) and isinstance(sym, str)
        key: tuple[str, str, str, int, int, float] = (
            model,
            fs,
            sym,
            train_size_int,
            seed_int,
            run_c_svc,
        )
        if key not in wanted or key in cache:
            continue
        cache[key] = ClassicalResult(
            model_name=model,
            accuracy=float(metrics.get(MetricKey.ACCURACY, 0.0)),
            balanced_accuracy=float(metrics.get(MetricKey.BALANCED_ACCURACY, 0.0)),
            mcc=float(metrics.get(MetricKey.MCC, 0.0)),
            f1=float(metrics.get(MetricKey.F1, 0.0)),
            precision=float(metrics.get(MetricKey.PRECISION, 0.0)),
            recall=float(metrics.get(MetricKey.RECALL, 0.0)),
            cm=None,
            y_pred=None,
            train_time_s=float(metrics.get(MetricKey.TRAIN_TIME_S, 0.0)),
            inference_time_s=float(metrics.get(MetricKey.INFERENCE_TIME_S, 0.0)),
            seed=seed_int,
            train_size=train_size_int,
            feature_set=fs,
            symmetry=sym,
            regime=reg,
            win_rate=metrics.get("win_rate"),
            c_svc=run_c_svc,
        )
    return cache


def log_classical_mlflow_run(result: ClassicalResult, mlflow: Any) -> None:
    """Log a single classical result to MLflow."""
    try:
        run_name = (
            f"{result.model_name}|{result.feature_set}|{result.symmetry}"
            f"|n={result.train_size}|s={result.seed}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    ParamKey.PIPELINE: PipelineValue.CLASSICAL,
                    ParamKey.MODEL: result.model_name,
                    ParamKey.FEATURE_SET: result.feature_set,
                    ParamKey.SYMMETRY: result.symmetry,
                    ParamKey.TRAIN_SIZE: result.train_size,
                    ParamKey.SEED: result.seed,
                    ParamKey.REGIME: result.regime,
                    ParamKey.C_SVC: float(result.c_svc),
                }
            )
            metrics: dict[str, float] = {
                MetricKey.ACCURACY: result.accuracy,
                MetricKey.BALANCED_ACCURACY: result.balanced_accuracy,
                MetricKey.MCC: result.mcc,
                MetricKey.F1: result.f1,
                MetricKey.PRECISION: result.precision,
                MetricKey.RECALL: result.recall,
                MetricKey.TRAIN_TIME_S: result.train_time_s,
                MetricKey.INFERENCE_TIME_S: result.inference_time_s,
            }
            if result.win_rate is not None:
                metrics[MetricKey.WIN_RATE] = result.win_rate
            mlflow.log_metrics(metrics)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        warnings.warn(f"MLflow logging failed: {exc}", stacklevel=2)
